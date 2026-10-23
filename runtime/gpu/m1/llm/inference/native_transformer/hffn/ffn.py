"""Route-grouped HFFN FFN: one up dispatch, down grouped (decode) or per route (prefill).

A plan is ((route, first_row, end_row), ...): contiguous row ranges covering
the batch, one route each. Route r executes banks (0, *pairs[r]) of the layer:
core + gate * (pair) + b_down, gate per row. Weights, indices and offsets stay
compact; activations are dense per executed bank.

Layouts (kernels.py): bank-major when every segment has n > 4 (each bank runs
once over the contiguous union of the segments using it; one input [d_model, N] and
two activation buffers: core bank and residual banks), route-major
otherwise. Down mode per segment:
cooperative (n <= 4), tiled, or blocked (two tokens per thread).

Aligned layout (aligned.py): in the bank-major layout the row strides of the input and of the activation
spans are rounded up to 16 floats (64 bytes), in layers whose FFN work per token reaches
aligned.MIN_WORK_PER_TOKEN and when the padded buffers keep their capacity; the outputs are
bitwise identical to the unaligned layout.

Resources:
- banks are loaded by `prepare(routes)` before a request; `forward` requires them;
- compiled libraries: three named LRU caches;
- shader metadata `ns`: fixed slots of one arena allocated at construction;
- workspaces: one input and one activation buffer per segment position,
  power-of-two capacities, bounded in total (`workspace_limit`, 64 MiB by default);
- lifetime: one record per FFN call retains every tensor the queued kernels
  read; at most `queue_depth` calls are in flight before the host waits.
"""
from collections import OrderedDict, deque

import torch

from .kernels import (regime, down_mode, down_groups, up_source, up_groups, down_source,
                      up_source_spans, up_groups_spans, down_source_spans, span_buffers, span_counts,
                      span_metadata, down_source_segment, SPAN_STRIDE, up_source_spans_runtime_widths,
                      ACTIVATIONS, with_activation, with_wide_indices)
from . import aligned as A

LIBRARY_CAPACITY = 2048
METADATA_SLOTS, METADATA_WIDTH = 2048, 4   # at most four routes, one N each
SPAN_WIDTH = 4*SPAN_STRIDE                 # bank-major span metadata, four segments
WORKSPACE_LIMIT = 64*1024**2    # default; a caller that runs more tokens gives its own to MoEFFN
QUEUE_DEPTH = 6
LAYOUT_CAPACITY = 4096                     # remembered layout decisions (one per call shape)


class MoEFFN:
    def __init__(self, store, output_biases, stats, queue_depth=QUEUE_DEPTH, workspace_limit=WORKSPACE_LIMIT):
        self.store, self.stats, self.queue_depth = store, stats, queue_depth
        self.workspace_limit = workspace_limit      # bytes, all workspaces together
        self.d_model = store.d_model                      # width of x
        self.d_out = getattr(store, 'd_out', store.d_model)    # width of y; the same unless the store says otherwise
        self.activation = getattr(store, 'activation', 'gelu')    # hidden activation of every bank of the store
        self.wide_indices = getattr(store, 'index_dtype', torch.int16) == torch.int32    # 16-bit unless the store says so
        if self.activation not in ACTIVATIONS:
            raise ValueError(f'unsupported activation: {self.activation}')
        self.routes = tuple((0, *pair) for pair in store.pairs)
        self.biases = tuple(b.to('mps') for b in output_biases)
        self.banks = {}
        self.bank_widths = None  # immutable per-layer views, allocated in first prepare
        self.up_libraries, self.decode_libraries, self.prefill_libraries = OrderedDict(), OrderedDict(), OrderedDict()
        self.arena = torch.zeros(METADATA_SLOTS*METADATA_WIDTH, dtype=torch.int32, device='mps')
        self.metadata, self.free_slots = OrderedDict(), list(range(METADATA_SLOTS-1, -1, -1))
        self.span_arena = torch.zeros(METADATA_SLOTS*SPAN_WIDTH, dtype=torch.int32, device='mps')
        self.span_metadata, self.free_span_slots = OrderedDict(), list(range(METADATA_SLOTS-1, -1, -1))
        self.workspaces = []
        self.layouts = {}
        self.aligned_layers = {}              # layer -> bool (FFN work per token)
        self.inflight = deque()
        self.closed = False

    # -- resources -------------------------------------------------------
    def prepare(self, routes):
        """Load, once, every bank the given routes use in every layer."""
        if len(self.routes) > 1 and self.bank_widths is None:
            widths = torch.tensor(self.store.widths, dtype=torch.int32, device='mps')
            self.bank_widths = tuple(widths[layer] for layer in range(self.store.layers))
        needed = sorted({b for r in routes for b in self.routes[r]})
        for layer in range(self.store.layers):
            for bank in needed:
                if (layer, bank) not in self.banks:
                    self.banks[layer, bank] = self.store.load(layer, bank)
                    self.stats.bank_loads += 1

    def _library(self, cache, key, source):
        library = cache.get(key)
        if library is None:
            text = with_activation(source(), self.activation)
            library = cache[key] = torch.mps.compile_shader(with_wide_indices(text) if self.wide_indices else text)
            self.stats.library_compiles += 1
            if len(cache) > LIBRARY_CAPACITY:
                cache.popitem(last=False)   # in-flight users keep their own reference
        else:
            cache.move_to_end(key)
        return library

    def _ns(self, ns):
        view = self.metadata.get(ns)
        if view is not None:
            self.metadata.move_to_end(ns)
            return view
        if not self.free_slots:
            evicted = self.metadata.popitem(last=False)[1]
            torch.mps.synchronize()          # no queued kernel may read the slot we rewrite
            self.free_slots.append(evicted.storage_offset()//METADATA_WIDTH)
            self.stats.metadata_evictions += 1
        slot = self.free_slots.pop()
        view = self.arena[slot*METADATA_WIDTH:slot*METADATA_WIDTH+len(ns)]
        view.copy_(torch.tensor(ns, dtype=torch.int32))
        self.metadata[ns] = view
        self.stats.metadata_writes += 1
        return view

    def _spans(self, segments, ns, rows_aligned=False):
        """Arena view holding the span metadata of the unaligned or of the aligned layout (same LRU
        policy as `_ns`)."""
        key = ('aligned', segments, ns) if rows_aligned else (segments, ns)
        view = self.span_metadata.get(key)
        if view is not None:
            self.span_metadata.move_to_end(key)
            return view
        if not self.free_span_slots:
            evicted = self.span_metadata.popitem(last=False)[1]
            torch.mps.synchronize()          # no queued kernel may read the slot we rewrite
            self.free_span_slots.append(evicted.storage_offset()//SPAN_WIDTH)
            self.stats.metadata_evictions += 1
        values = A.span_metadata(segments, ns) if rows_aligned else span_metadata(segments, ns)
        slot = self.free_span_slots.pop()
        view = self.span_arena[slot*SPAN_WIDTH:slot*SPAN_WIDTH+len(values)]
        view.copy_(torch.tensor(values, dtype=torch.int32))
        self.span_metadata[key] = view
        self.stats.metadata_writes += 1
        return view

    def _workspace(self, position, inputs, activations):
        while len(self.workspaces) <= position:
            self.workspaces.append(None)
        old = self.workspaces[position]
        if old is None or old[0].numel() < inputs or old[1].numel() < activations:
            sizes = [1 << (k-1).bit_length() for k in (inputs, activations)]
            if old is not None:
                sizes = [max(k, t.numel()) for k, t in zip(sizes, old)]
            others = sum(t.numel() for i, w in enumerate(self.workspaces) if w and i != position for t in w)
            if 4*(sum(sizes)+others) > self.workspace_limit:
                raise RuntimeError('FFN workspaces would exceed their limit')
            self.workspaces[position] = tuple(torch.empty(k, device='mps') for k in sizes)
            self.stats.workspace_allocations += 1
        return self.workspaces[position]

    def _aligned_layer(self, layer):
        """True when this layer's FFN work per token reaches aligned.MIN_WORK_PER_TOKEN (static per layer)."""
        decision = self.aligned_layers.get(layer)
        if decision is None:
            edges = {b: int(self.banks[layer, b].up_index.numel())+int(self.banks[layer, b].down_hidden.numel())
                     for b in sorted({b for route in self.routes for b in route})}
            decision = self.aligned_layers[layer] = A.worth_aligning(A.layer_work(edges, self.routes))
        return decision

    def _aligned_sizes(self, layer, segments, ns):
        """Activation buffer sizes of the aligned layout of a bank-major call, or None when the call
        keeps the unaligned layout (light layer, already aligned, or a padded work buffer would grow)."""
        if not self._aligned_layer(layer):
            return None
        if self.bank_widths is None:          # one route, one bank
            n, hidden = ns[0], segments[0][1][0]
            return {0: hidden*A.aligned(n)} if len(ns) == 1 and A.one_segment_aligned(n, hidden, self.d_model) else None
        if all(n % A.ALIGN == 0 for n in ns):
            return None
        key = (segments, ns)
        sizes = self.layouts.get(key)
        if sizes is None:                     # depends on the call's shape only
            if len(self.layouts) >= LAYOUT_CAPACITY:
                self.layouts.clear()
            sizes = self.layouts[key] = A.span_buffers(segments, ns) if A.spans_aligned(segments, ns, self.d_model) else False
        return sizes or None

    def _admit(self):
        while self.inflight and self.inflight[0][0].query():
            self.inflight.popleft()
        while len(self.inflight) >= self.queue_depth:
            self.inflight.popleft()[0].synchronize()
            self.stats.queue_waits += 1

    def _retain(self, tensors):
        event = torch.mps.Event()
        event.record()
        self.inflight.append((event, tensors))
        self.stats.queue_peak = max(self.stats.queue_peak, len(self.inflight))

    # -- execution -------------------------------------------------------
    def forward(self, layer, x, plan, gates, phase):
        if self.closed:
            raise RuntimeError('FFN closed')
        if phase not in ('prefill', 'decode'):
            raise ValueError('invalid phase')
        rows, steps, width = x.shape
        d_model = self.d_model
        if width != d_model or x.dtype != torch.float32 or gates.shape != (rows,):
            raise ValueError(f'expected FP32 [rows, steps, {d_model}] input and one gate per row')
        if plan[0][1] != 0 or plan[-1][2] != rows or any(p[2] != q[1] for p, q in zip(plan, plan[1:])):
            raise ValueError('plan must cover the rows contiguously')
        widths = self.store.widths[layer]
        segments = tuple((self.routes[r], tuple(widths[b] for b in self.routes[r])) for r, _, _ in plan)
        ns = tuple((end-first)*steps for _, first, end in plan)
        modes = tuple(down_mode(n, sum(bank_widths)) for n, (_, bank_widths) in zip(ns, segments))
        bank_major = min(ns) > 4
        banks = sorted({b for bank_ids, _ in segments for b in bank_ids})
        stored = {b: self.banks[layer, b] for b in banks}    # KeyError: banks were not prepared

        self._admit()
        metadata = self._ns(ns)
        up_args = [metadata]
        for b in banks:
            up_args += [stored[b].up_offsets, stored[b].up_index, stored[b].up_weights]
        rows_aligned = False
        if bank_major:
            # Regimes and down modes are chosen at run time in bank-major kernels, so their libraries are keyed
            # by the plan structure only (fewer pipelines, less driver memory). Row strides are rounded up to
            # 16 floats when the padded buffers keep their capacity.
            sizes = self._aligned_sizes(layer, segments, ns)
            rows_aligned = sizes is not None
            if self.bank_widths is not None:
                # Layout and grid stay host-owned. The immutable actual widths
                # let layers share byte-identical up sources for the same bank plan.
                if rows_aligned:
                    up = self._library(self.up_libraries, ('aligned_bank_runtime_widths', tuple(s[0] for s in segments)),
                                       lambda: A.up_source(segments))
                else:
                    up = self._library(self.up_libraries, ('bank_runtime_widths', tuple(s[0] for s in segments)),
                                       lambda: up_source_spans_runtime_widths(segments))
                up_args = [up_args[0], self.bank_widths[layer], *up_args[1:]]
            elif rows_aligned:
                up = self._library(self.up_libraries, ('aligned_bank', segments),
                                   lambda: A.one_segment_source('up', up_source_spans(segments)))
            else:
                up = self._library(self.up_libraries, ('bank', segments), lambda: up_source_spans(segments))
            total = rows*steps
            stride = A.aligned(total) if rows_aligned else total
            if not rows_aligned:
                sizes = span_buffers(segments, ns)
            buffers = []
            for k, size in sizes.items():   # workspace k: (input X for k == 0, activation buffer Ak)
                inputs_k, acts_k = self._workspace(k, d_model*stride if k == 0 else 1, size)
                if k == 0:
                    inputs = inputs_k[:d_model*stride].view(d_model, stride)
                buffers.append(acts_k[:size])
            if rows_aligned:    # the same values at the same addresses as inputs[:, :total] <- x.T;
                inputs.T[:total].copy_(x.reshape(total, d_model))   # the other columns are never read
            else:
                inputs.copy_(x.reshape(total, d_model).T)
            up_args += [inputs, *buffers]
            groups = up_groups_spans(segments, ns)
            activations = lambda positions: buffers
            self.stats.bank_major_calls += 1
        else:
            regimes = tuple(regime(n) for n in ns)
            up = self._library(self.up_libraries, ('route', segments, regimes),
                               lambda: up_source(segments, regimes, d_model=d_model))
            per_segment = []
            for g, ((_, first, end), (_, bank_widths), n) in enumerate(zip(plan, segments, ns)):
                inputs, acts = self._workspace(g, d_model*n, sum(bank_widths)*n)
                source = x[first:end].reshape(n, d_model)
                inputs = inputs[:d_model*n].view((n, d_model) if n <= 4 else (d_model, n))
                inputs.copy_(source if n <= 4 else source.T)
                acts = acts[:sum(bank_widths)*n]
                up_args += [inputs, acts]
                per_segment.append(acts)
            groups = up_groups(segments, ns)
            activations = lambda positions: [per_segment[g] for g in positions]
            self.stats.route_major_calls += 1
        generate = down_source_spans if bank_major else down_source
        layout = 'bank' if bank_major else 'route'
        d_out = self.d_out
        out = torch.empty_like(x) if d_out == d_model else x.new_empty((rows, steps, d_out))
        bias = self.biases[layer]
        up.up(*up_args, threads=groups*512, group_size=512)
        self.stats.up_dispatches += 1
        retained = [up, x, gates, out, *up_args]

        def down_args(positions):
            args = [metadata]
            for b in sorted({b for g in positions for b in segments[g][0]}):
                args += [stored[b].down_offsets, stored[b].down_hidden, stored[b].down_values]
            return args + activations(positions) + [gates, bias, out]

        if phase == 'decode':
            key_modes = ('dynamic',)*len(ns) if bank_major else modes
            if rows_aligned and self.bank_widths is not None:
                down = self._library(self.decode_libraries, ('aligned_bank', segments, key_modes),
                                     lambda: A.down_source(segments, key_modes, d_model=d_out))
            elif rows_aligned:
                down = self._library(self.decode_libraries, ('aligned_bank', segments, key_modes),
                                     lambda: A.one_segment_source(
                                         'down', down_source_spans(segments, key_modes, d_model=d_out)))
            else:
                down = self._library(self.decode_libraries, (layout, segments, key_modes),
                                     lambda: generate(segments, key_modes, d_model=d_out))
            args = down_args(range(len(ns)))
            down.down(*args, steps, threads=sum(down_groups(n, m, d_out) for n, m in zip(ns, modes))*512, group_size=512)
            self.stats.down_dispatches += 1
            retained += [down, *args]
        else:
            spans = self._spans(segments, ns, rows_aligned) if bank_major else None
            for g, (n, mode) in enumerate(zip(ns, modes)):
                if bank_major:   # library keyed by this segment only; span layout read from `spans`
                    kernel_mode = 'dynamic'
                    down = self._library(self.prefill_libraries, ('bank', g, segments[g], kernel_mode),
                                         lambda: down_source_segment(segments[g], kernel_mode, g, d_model=d_out))
                    args = down_args((g,))
                    args = [args[0], spans, *args[1:]]
                else:
                    down = self._library(self.prefill_libraries, (layout, g, segments, mode),
                                         lambda: generate(segments, (mode,), positions=(g,), d_model=d_out))
                    args = down_args((g,))
                down.down(*args, steps, threads=down_groups(n, mode, d_out)*512, group_size=512)
                self.stats.down_dispatches += 1
                retained += [down, *args]
        self._retain(tuple(retained))
        self.stats.ffn_calls += 1
        return out

    def close(self):
        if not self.closed:
            torch.mps.synchronize()
            self.inflight.clear()
            for cache in (self.banks, self.up_libraries, self.decode_libraries, self.prefill_libraries, self.metadata,
                          self.span_metadata):
                cache.clear()
            self.workspaces.clear()
            self.layouts.clear()
            self.aligned_layers.clear()
            self.arena = self.span_arena = None
            self.bank_widths = None
            self.closed = True
