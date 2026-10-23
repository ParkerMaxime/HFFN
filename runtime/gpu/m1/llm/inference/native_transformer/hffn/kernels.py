"""Metal source generators for route-grouped hidden-major FFN execution.

A call is a list of route segments (contiguous token ranges). Each segment
executes a core bank plus zero to two residual banks; its token count N is read
at run time from `ns[g]`, only the kernel regimes and down modes are fixed at
compile time.

Two layouts (see ffn.py):
- bank-major (every segment has n > 4): each bank runs once over the contiguous
  union of the segments that use it (a span); one input X [d_model, sum ns] and two
  activation buffers: A0 holds the core bank's span, A1 every residual span
  (one buffer per span would exceed Metal's 31 buffer arguments);
- route-major (some segment has n <= 4): one input and one activation buffer per
  segment.

up    z = b_up + <w_up, x>, a = activation(z) (gelu in the generated text, see with_activation),
      hidden-major a[hidden*n + token].
      Regime 0 (n <= 4, route-major only): SIMD reduction per token. Regimes 1
      (lane per token) and 2 (128-token tiles, four accumulators) accumulate in the
      same order: edges four at a time (loads first), then the remainder.
down  output-owner reduction per segment:
      y = core + gate * (residual_1 [+ residual_2]) + b_down   (core only: y = core + b_down)
      modes: 'coop' (n <= 4, SIMD reduction), 'tiled' (thread per (output, token),
      grid in tiles of 64 tokens), 'blocked' (two tokens per thread, lane l owns tokens
      l and l+32 of a 64-token tile). Edges four at a time, same order in every mode.
      Decode: one dispatch for all segments; prefill: one dispatch per segment.

Rounding: per-bank sums in edge order, then the gated sum and the bias with
`volatile` stores so the compiler cannot contract them into FMAs. 

d_model (the width of x and y) is a compile-time constant of each generated library, given by the
caller; called with d_model = 256, every generator emits the source of the fixed width 256.
"""

PRELUDE = '''
inline float erf_accurate(float x) {
  float ax=metal::abs(x),t=1.0f/(1.0f+0.3275911f*ax);
  float p=(((((1.061405429f*t-1.453152027f)*t+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t);
  float y=1.0f-p*metal::exp(-ax*ax);
  return x<0.0f?-y:y;
}
inline float gelu(float x) { return 0.5f*x*(1.0f+erf_accurate(x*0.7071067811865475f)); }
'''
ACTIVATIONS = {    # hidden activation: its Metal definition; the exact-erf GELU is the one of PRELUDE
    'gelu': None,
    'relu': 'inline float relu(float x) { return metal::max(x, 0.0f); }\n',
}
IDS = 'uint tid[[thread_position_in_threadgroup]],uint tg[[threadgroup_position_in_grid]]'


def with_wide_indices(source):
    """A generated source whose compact indices (input features `i`, hidden units `h`) are 32-bit instead of
    16-bit. Offsets are 32-bit in both cases."""
    wide = source.replace('device const short* i', 'device const int* i').replace('device const short* h', 'device const int* h')
    if 'short' in wide:
        raise ValueError('a 16-bit declaration was not rewritten')
    return wide


def with_activation(source, activation):
    """A generated source with its hidden activation replaced; `gelu` returns the source itself."""
    definition = ACTIVATIONS[activation]
    if definition is None:
        return source
    if not source.startswith(PRELUDE):
        raise ValueError('not a source of this module')
    return PRELUDE + definition + source[len(PRELUDE):].replace('=gelu(', f'={activation}(')
MAX_ARGS = 31
UNROLL = 4          # edges per loop iteration
TILE = 64           # tokens per down grid tile
LANES = 32
BLOCK = 2           # tokens per thread in the blocked down
BLOCK_MAX_WIDTH = 2560   # blocked down only for segments with a narrower bank sum, or n <= 128


def regime(n):
    """Up kernel regime: SIMD reduction per token, lane per token, 128-token tiles."""
    return 0 if n <= 4 else 1 if n < 64 else 2


def down_mode(n, width):
    """Down mode of a segment of n tokens whose banks sum to `width` hidden units."""
    if n <= 4:
        return 'coop'
    if n >= LANES*BLOCK and (width <= BLOCK_MAX_WIDTH or n <= 2*LANES*BLOCK):
        return 'blocked'
    return 'tiled'


def output_groups(d_model):
    """Threadgroups of 512 threads covering the d_model outputs once (16 SIMD groups of 32 lanes)."""
    return (d_model+15)//16


def down_groups(n, mode, d_model):
    """Threadgroups of 512 threads for one segment."""
    if mode == 'coop':
        return output_groups(d_model)
    if mode == 'blocked':
        return output_groups(d_model)*((n+LANES*BLOCK-1)//(LANES*BLOCK))
    return (n*d_model+511)//512


def _banks(segments):
    return sorted({b for banks, _ in segments for b in banks})


def _sum(first, end):
    terms = [f'ns[{p}]' for p in range(first, end)]
    return '('+'+'.join(terms)+')' if terms else '0u'


# -- bank-major layout ------------------------------------------------------------
def spans_of(segments):
    """((bank, first position, end position), ...): maximal runs of consecutive segments using a bank."""
    out = []
    for b in _banks(segments):
        g = 0
        while g < len(segments):
            if b in segments[g][0]:
                first = g
                while g < len(segments) and b in segments[g][0]:
                    g += 1
                out.append((b, first, g))
            else:
                g += 1
    return tuple(out)


def span_layout(segments):
    """Per span: bank, width, positions, token count, token origin, activation buffer (0: core bank,
    1: residual banks) and offset in that buffer (Metal expressions)."""
    width = {b: h for banks, widths in segments for b, h in zip(banks, widths)}
    rows, offsets = [], {0: '0u', 1: '0u'}
    for b, first, end in spans_of(segments):
        count, buffer = _sum(first, end), 0 if b == 0 else 1
        rows.append(dict(bank=b, width=width[b], first=first, end=end, n=count, origin=_sum(0, first),
                         buffer=buffer, offset=offsets[buffer]))
        offsets[buffer] = f'({offsets[buffer]}+{width[b]}u*{count})'
    return rows


def span_counts(segments, ns):
    return tuple(sum(ns[first:end]) for _, first, end in spans_of(segments))


def span_buffers(segments, ns):
    """{buffer: floats} for the activation buffers of a bank-major call."""
    sizes = {}
    for span, n in zip(span_layout(segments), span_counts(segments, ns)):
        sizes[span['buffer']] = sizes.get(span['buffer'], 0) + span['width']*n
    return dict(sorted(sizes.items()))


# -- up ---------------------------------------------------------------------------
def _up_body(b, h, n, reg, x, store):
    """One bank block of the up for regimes 1 and 2. x(i, t): input address; store(r, t): output address."""
    bias = f'w{b}[r]'
    W = lambda e: f'w{b}[{h}+{e}]'
    I = lambda e: f'uint(i{b}[{e}])'
    loads = ''.join(f'uint ii{u}={I(f"e+{u}")};float ww{u}={W(f"e+{u}")};' for u in range(UNROLL))
    if reg == 1:
        loads += ''.join(f'float xx{u}={x(f"ii{u}", "t")};' for u in range(UNROLL))
        acc = ''.join(f's+=ww{u}*xx{u};' for u in range(UNROLL))
        return (f'for(uint t=lane;t<{n};t+=32){{float s={bias};uint e=begin;'
                f'for(;e+{UNROLL}<=end;e+={UNROLL}){{{loads}{acc}}}'
                f'for(;e<end;++e)s+={W("e")}*{x(I("e"), "t")};'
                f'{store("r", "t")}=gelu(s);}}')
    acc = ''.join(f'if(t{k}<{n}){{' + ''.join(f's{k}+=ww{u}*{x(f"ii{u}", f"t{k}")};' for u in range(UNROLL)) + '}'
                  for k in range(4))
    tail = ''.join(f'if(t{k}<{n})s{k}+=w*{x("i", f"t{k}")};' for k in range(4))
    return (f'uint t0=tile*128+lane,t1=t0+32,t2=t0+64,t3=t0+96;float s0={bias},s1=s0,s2=s0,s3=s0;uint e=begin;'
            f'for(;e+{UNROLL}<=end;e+={UNROLL}){{{loads}{acc}}}'
            f'for(;e<end;++e){{uint i={I("e")};float w={W("e")};{tail}}}'
            + ''.join(f'if(t{k}<{n}){store("r", f"t{k}")}=gelu(s{k});' for k in range(4)))


def _up_block(b, h, n, reg, start):
    count = (h+15)//16
    total = f'({count}*(({n}+127)/128))' if reg == 2 else f'({count}*1)'
    head = (f'uint z=tg-{start},tile=z/{count},r=(z%{count})*16+simd;if(r>={h})return;'
            f'uint begin=uint(o{b}[r]),end=uint(o{b}[r+1]);')
    return head, total


def up_source(segments, regimes, *, d_model):
    """Route-major up. segments: ((bank ids, widths), ...); weights buffer w{b} = [b_up(width) | up values]."""
    args = ['constant uint* ns']
    for b in _banks(segments):
        args += [f'device const int* o{b}', f'device const short* i{b}', f'device const float* w{b}']
    for g in range(len(segments)):
        args += [f'device const float* x{g}', f'device float* a{g}']
    assert len(args) <= MAX_ARGS
    code, start = 'uint lane=tid&31u,simd=tid>>5u;', '0'
    for g, ((banks, widths), reg) in enumerate(zip(segments, regimes)):
        n, local = f'ns[{g}]', '0'
        for b, h in zip(banks, widths):
            head, total = _up_block(b, h, n, reg, start)
            if reg == 0:
                body = (f'for(uint t=0;t<{n};++t){{float p=0;for(uint e=begin+lane;e<end;e+=32)'
                        f'p+=w{b}[{h}+e]*x{g}[t*{d_model}+uint(i{b}[e])];float s=metal::simd_sum(p);'
                        f'if(lane==0)a{g}[({local})+r*({n})+t]=gelu(w{b}[r]+s);}}')
            else:
                body = _up_body(b, h, n, reg, lambda i, t: f'x{g}[{i}*{n}+{t}]',
                                lambda r, t, local=local: f'a{g}[({local})+{r}*({n})+{t}]')
            code += f'if(tg>={start} && tg<({start}+{total})){{{head}{body}}}'
            start, local = f'({start}+{total})', f'({local}+{h}*{n})'
    return PRELUDE + f'kernel void up({",".join(args)},{IDS}){{{code}}}'


def up_groups(segments, ns):
    """Route-major up threadgroups of 512 threads (16 SIMD groups, one hidden unit each)."""
    return sum(((w+15)//16)*((n+127)//128 if n >= 64 else 1)
               for (banks, widths), n in zip(segments, ns) for w in widths)


def up_source_spans(segments, regimes=None):
    """Bank-major up; regimes: one per span (1 or 2), or None: regime 1 or 2 chosen at run
    time from the span's token count, both bodies in one library. Input X [d_model, sum ns]."""
    args = ['constant uint* ns']
    for b in _banks(segments):
        args += [f'device const int* o{b}', f'device const short* i{b}', f'device const float* w{b}']
    args += ['device const float* X'] + [f'device float* A{k}' for k in sorted({s['buffer'] for s in span_layout(segments)})]
    assert len(args) <= MAX_ARGS
    total_tokens = _sum(0, len(segments))
    code, start = 'uint lane=tid&31u,simd=tid>>5u;', '0'
    layout = span_layout(segments)
    for span, reg in zip(layout, regimes or (None,)*len(layout)):
        assert reg in (1, 2, None)
        b, h, n = span['bank'], span['width'], span['n']
        x = lambda i, t, o=span['origin']: f'X[{i}*{total_tokens}+{o}+{t}]'
        store = lambda r, t, off=span['offset'], k=span['buffer']: f'A{k}[{off}+{r}*{n}+{t}]'
        if reg is None:
            head, total_2 = _up_block(b, h, n, 2, start)
            _, total_1 = _up_block(b, h, n, 1, start)
            total = f'({n}<64u?{total_1}:{total_2})'
            body = f'if({n}<64u){{{_up_body(b, h, n, 1, x, store)}}}else{{{_up_body(b, h, n, 2, x, store)}}}'
        else:
            head, total = _up_block(b, h, n, reg, start)
            body = _up_body(b, h, n, reg, x, store)
        code += f'if(tg>={start} && tg<({start}+{total})){{{head}{body}}}'
        start = f'({start}+{total})'
    return PRELUDE + f'kernel void up({",".join(args)},{IDS}){{{code}}}'


def up_groups_spans(segments, ns):
    return sum(((s['width']+15)//16)*((n+127)//128 if n >= 64 else 1)
               for s, n in zip(span_layout(segments), span_counts(segments, ns)))


# -- down -------------------------------------------------------------------------
def _blocked_condition(n, width):
    """Metal expression equal to `down_mode(n, width) == 'blocked'` for n > 4 (width is compile-time)."""
    if width <= BLOCK_MAX_WIDTH:
        return f'({n}>={LANES*BLOCK}u)'
    return f'({n}>={LANES*BLOCK}u && {n}<={2*LANES*BLOCK}u)'


def _down_segment(g, banks, n, base, mode, first, act, width, d_model):
    """Code block of one segment. mode 'dynamic' (n > 4 only): tiled and blocked bodies in one
    library, the mode chosen at run time from n exactly as `down_mode` does on the host."""
    if mode == 'dynamic':
        blocked = _blocked_condition(n, width)
        body_b, count_b = _down_body(g, banks, n, base, 'blocked', first, act, d_model)
        body_t, count_t = _down_body(g, banks, n, base, 'tiled', first, act, d_model)
        count = f'({blocked}?{count_b}:{count_t})'
        return f'if(tg>=({first}) && tg<(({first})+({count}))){{if({blocked}){{{body_b}}}else{{{body_t}}}}}', count
    body, count = _down_body(g, banks, n, base, mode, first, act, d_model)
    return f'if(tg>=({first}) && tg<(({first})+({count}))){{{body}}}', count


def _down_body(g, banks, n, base, mode, first, act, d_model):
    """Body and threadgroup count of one segment. act(b, q, token): activation address of edge q of bank b."""
    groups = output_groups(d_model)
    count = f'{groups}' if mode == 'coop' else (f'({groups}u*(({n}+{LANES*BLOCK-1}u)/{LANES*BLOCK}u))' if mode == 'blocked'
                                                else f'(({n}*{d_model}+511)/512)')
    if mode == 'coop':
        body = f'uint lane=tid&31u,o=(tg-({first}))*16u+(tid>>5u);if(o>={d_model})return;for(uint n=0;n<{n};++n){{'
        for j, b in enumerate(banks):
            body += (f'float p{j}=0;for(uint q=uint(o{b}[o])+lane;q<uint(o{b}[o+1]);q+=32)p{j}+=v{b}[q]*{act(b, "q", "n")};'
                     f'float s{j}=metal::simd_sum(p{j});')
        tokens, sums = [('n', [f's{j}' for j in range(len(banks))])], None
    elif mode == 'tiled':
        body = (f'uint gid=(tg-({first}))*512+tid;if(gid>={n}*{d_model})return;'
                f'uint tt=gid/({d_model}u*{TILE}u),rr=gid%({d_model}u*{TILE}u),mm=metal::min({TILE}u,{n}-tt*{TILE}u);'
                f'uint o=rr/mm,n=tt*{TILE}u+rr%mm;{{')
        for j, b in enumerate(banks):
            loads = ''.join(f'float vv{u}=v{b}[q+{u}];float aa{u}={act(b, f"q+{u}", "n")};' for u in range(UNROLL))
            acc = ''.join(f's{j}+=vv{u}*aa{u};' for u in range(UNROLL))
            body += (f'float s{j}=0;{{uint q=uint(o{b}[o]),qe=uint(o{b}[o+1]);'
                     f'for(;q+{UNROLL}<=qe;q+={UNROLL}){{{loads}{acc}}}'
                     f'for(;q<qe;++q)s{j}+=v{b}[q]*{act(b, "q", "n")};}}')
        tokens = [('n', [f's{j}' for j in range(len(banks))])]
    else:
        tile = LANES*BLOCK
        body = (f'uint gid=(tg-({first}))*512+tid,tl=gid/({d_model}u*32u),rr=gid%({d_model}u*32u),'
                f'o=rr/32u,lane=rr%32u,tb=tl*{tile}u+lane;if(tb>={n})return;'
                f'bool full=tl*{tile}u+{tile}u<={n};{{')
        for j, b in enumerate(banks):
            def loop(checked, b=b, j=j):
                guard = (lambda k, s: f'if(tb+{LANES*k}u<{n}){{{s}}}') if checked else (lambda k, s: s)
                loads = ''.join(f'float vv{u}=v{b}[q+{u}];' for u in range(UNROLL))
                acc = ''.join(guard(k, ''.join(f'float aa{u}_{k}={act(b, f"q+{u}", f"tb+{LANES*k}u")};'
                                               for u in range(UNROLL))
                                    + ''.join(f's{j}_{k}+=vv{u}*aa{u}_{k};' for u in range(UNROLL)))
                              for k in range(BLOCK))
                tail = ''.join(guard(k, f's{j}_{k}+=v{b}[q]*{act(b, "q", f"tb+{LANES*k}u")};') for k in range(BLOCK))
                return (f'{{uint q=uint(o{b}[o]),qe=uint(o{b}[o+1]);'
                        f'for(;q+{UNROLL}<=qe;q+={UNROLL}){{{loads}{acc}}}for(;q<qe;++q){{{tail}}}}}')
            body += ''.join(f'float s{j}_{k}=0;' for k in range(BLOCK)) + f'if(full){loop(False)}else{loop(True)}'
        tokens = [(f'tb+{LANES*k}u', [f's{j}_{k}' for j in range(len(banks))]) for k in range(BLOCK)]
    writes = ''
    for k, (local, sums) in enumerate(tokens):
        token = f'(({base})+{local})'
        if len(banks) > 1:
            write = (f'volatile float sum{k}={"+".join(sums[1:])};'
                     f'volatile float scaled{k}=gates[{token}/steps]*sum{k};volatile float core{k}={sums[0]}+scaled{k};'
                     f'volatile float answer{k}=core{k}+bias[o];y[{token}*{d_model}+o]=answer{k};')
        else:
            write = f'volatile float answer{k}={sums[0]}+bias[o];y[{token}*{d_model}+o]=answer{k};'
        if mode == 'coop':
            write = f'if(lane==0){{{write}}}'
        elif mode == 'blocked':
            write = f'if({local}<{n}){{{write}}}'
        writes += write
    body += writes + '}'
    return body, count


def _down_kernel(segments, modes, positions, act_args, act_of, d_model):
    positions = list(range(len(segments))) if positions is None else list(positions)
    args = ['constant uint* ns']
    for b in _banks([segments[g] for g in positions]):
        args += [f'device const int* o{b}', f'device const short* h{b}', f'device const float* v{b}']
    args += act_args(positions)
    args += ['device const float* gates', 'device const float* bias', 'device float* y', 'constant uint& steps']
    assert len(args) <= MAX_ARGS
    code, first = '', '0'
    for g, mode in zip(positions, modes):
        base = '+'.join(f'ns[{p}]' for p in range(g)) or '0'
        block, count = _down_segment(g, segments[g][0], f'ns[{g}]', base, mode, first, act_of(g), sum(segments[g][1]),
                                     d_model)
        code += block
        first = f'({first}+{count})'
    return PRELUDE + f'kernel void down({",".join(args)},{IDS}){{{code}}}'


def down_source(segments, modes, positions=None, *, d_model):
    """Route-major down: activations a{g} = [bank hidden, n] per segment."""
    def act_of(g):
        offsets, total = {}, 0
        for b, h in zip(*segments[g]):
            offsets[b], total = total, total+h
        return lambda b, q, t: f'a{g}[({offsets[b]}+uint(h{b}[{q}]))*ns[{g}]+{t}]'
    return _down_kernel(segments, modes, positions, lambda ps: [f'device const float* a{g}' for g in ps], act_of,
                        d_model)


def down_source_spans(segments, modes, positions=None, *, d_model):
    """Bank-major down: each bank's activations are read from its span in A."""
    span_of = {(s['bank'], g): s for s in span_layout(segments) for g in range(s['first'], s['end'])}

    def act_of(g):
        def act(b, q, t):
            s = span_of[b, g]
            return f'A{s["buffer"]}[({s["offset"]})+uint(h{b}[{q}])*{s["n"]}+{_sum(s["first"], g)}+{t}]'
        return act
    buffers = sorted({s['buffer'] for s in span_layout(segments)})
    return _down_kernel(segments, modes, positions, lambda ps: [f'device const float* A{k}' for k in buffers], act_of,
                        d_model)


# -- bank-major prefill down with runtime span metadata --------------------------
SPAN_STRIDE = 9     # per segment: (offset, span tokens, token delta) for up to three banks


def span_metadata(segments, ns):
    """Flat uint values read by `down_source_segment`: for each segment g and each bank j of
    its route, the span's offset in its activation buffer, the span's token count and the
    segment's token delta inside the span (zero-padded to SPAN_STRIDE per segment)."""
    offsets, spans = {0: 0, 1: 0}, {}
    for s, n in zip(span_layout(segments), span_counts(segments, ns)):
        spans[s['bank'], s['first']] = (offsets[s['buffer']], n, s['first'], s['end'])
        offsets[s['buffer']] += s['width']*n
    out = []
    for g, (banks, _) in enumerate(segments):
        row = []
        for b in banks:
            offset, n, first, end = next(v for (bank, f), v in spans.items() if bank == b and v[2] <= g < v[3])
            row += [offset, n, sum(ns[first:g])]
        out += row + [0]*(SPAN_STRIDE-len(row))
    return tuple(out)


def down_source_segment(segment, mode, g, *, d_model):
    # called with mode 'dynamic' (tiled/blocked chosen at run time)
    """Bank-major down for one segment at plan position g (prefill). Span offsets, counts and
    deltas come from `sm` at run time, so the library depends only on the segment's banks,
    widths, mode and position, not on the rest of the plan."""
    banks, _ = segment
    buffers = sorted({0 if b == 0 else 1 for b in banks})
    args = ['constant uint* ns', 'constant uint* sm']
    for b in sorted(banks):
        args += [f'device const int* o{b}', f'device const short* h{b}', f'device const float* v{b}']
    args += [f'device const float* A{k}' for k in buffers]
    args += ['device const float* gates', 'device const float* bias', 'device float* y', 'constant uint& steps']
    assert len(args) <= MAX_ARGS
    slot = {b: SPAN_STRIDE*g+3*j for j, b in enumerate(banks)}

    def act(b, q, t):
        k = slot[b]
        return f'A{0 if b == 0 else 1}[sm[{k}]+uint(h{b}[{q}])*sm[{k+1}]+sm[{k+2}]+{t}]'
    base = '+'.join(f'ns[{p}]' for p in range(g)) or '0'
    block, _ = _down_segment(g, banks, f'ns[{g}]', base, mode, '0', act, sum(segment[1]), d_model)
    return PRELUDE + f'kernel void down({",".join(args)},{IDS}){{{block}}}'


# -- immutable up widths at runtime --
def up_source_spans_runtime_widths(segments):
    """Bank-major up shared across MoE layers; immutable bw holds actual bank widths."""
    args=['constant uint* ns','constant uint* bw']
    for b in _banks(segments):
        args += [f'device const int* o{b}',f'device const short* i{b}',f'device const float* w{b}']
    args += ['device const float* X']+[f'device float* A{k}' for k in sorted({s['buffer'] for s in span_layout(segments)})]
    assert len(args)<=MAX_ARGS
    total_tokens=_sum(0,len(segments))
    code,start='uint lane=tid&31u,simd=tid>>5u;','0u'
    # Width-dependent span offsets are runtime integer expressions; span order
    # and the per-neuron active-edge sum retain the frozen generator's order.
    offsets={0:'0u',1:'0u'}
    for span in span_layout(segments):
        b,n,k=span['bank'],span['n'],span['buffer']
        h=f'bw[{b}]'; count=f'(({h}+15u)/16u)'
        total=f'({count}*({n}<64u?1u:(({n}+127u)/128u)))'
        head=(f'uint z=tg-{start},tile=z/{count},r=(z%{count})*16u+simd;if(r>={h})return;'
              f'uint begin=uint(o{b}[r]),end=uint(o{b}[r+1]);')
        x=lambda i,t,o=span['origin']:f'X[{i}*{total_tokens}+{o}+{t}]'
        store=lambda r,t,off=offsets[k],k=k:f'A{k}[{off}+{r}*{n}+{t}]'
        body=f'if({n}<64u){{{_up_body(b,h,n,1,x,store)}}}else{{{_up_body(b,h,n,2,x,store)}}}'
        code+=f'if(tg>={start} && tg<({start}+{total})){{{head}{body}}}'
        start=f'({start}+{total})'
        offsets[k]=f'({offsets[k]}+{h}*{n})'
    return PRELUDE+f'kernel void up({",".join(args)},{IDS}){{{code}}}'

