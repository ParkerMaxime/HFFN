"""Activation row strides aligned on 64 bytes, for the bank-major layout.

The unaligned kernels address X as `feature*N + origin + token` (N: tokens of the call) and each bank span of A
as `offset + hidden*n + token` (n: tokens of the span). Here both strides are rounded up to 16 floats and span
offsets accumulate on the rounded strides. Span origins in X, segment positions inside a span, grids, threads,
token-to-lane mapping, edge order and summation order are unchanged: outputs are bitwise identical to the
unaligned layout. Weights, indices and offsets are untouched; no support is padded.

The aligned layout is used only in layers whose FFN work per token reaches MIN_WORK_PER_TOKEN, and only when
every padded work buffer keeps the capacity (power of two) the unaligned layout would reserve, so no work
buffer grows. Calls whose segments are all multiples of 16 tokens are already aligned and keep the unaligned
layout, as do route-major calls.

The sources are the one-segment text of the unaligned kernels with its stride sites rewritten, and the generators
with a stride parameter. The prefill down is the unaligned library, unmodified: its stride and offsets come from the
span metadata.
"""
from . import kernels as K

ALIGN = 16                                  # floats per 64-byte line
ONE_SEGMENT_SITE = '*(ns[0])+'              # every address stride of the one-segment kernels
ONE_SEGMENT_SITES = dict(up=30, down=25)    # occurrences in the one-segment sources (checked)

# Below this FFN work per token and layer (active up + down connections, MAC), the expected kernel saving of a
# prefill-sized call does not exceed three times the fixed host cost of the aligned path. Estimate: kernel cost
# 24 ps per MAC at about 128 tokens, saving 12 % of it, fixed cost 8 µs per call: 3 x 8 µs / (0.12 x 24 ps x
# 128 tokens) = 65 104 MAC per token, rounded to 2**16.
MIN_WORK_PER_TOKEN = 1 << 16


def aligned(n):
    return (n+ALIGN-1)//ALIGN*ALIGN


def aligned_expression(n):
    """Metal expression of `aligned(n)` for a token-count expression n."""
    return f'(({n}+{ALIGN-1}u)&~{ALIGN-1}u)'


def published_expression(n):
    return n


def layer_work(edges, routes):
    """FFN work per token of one layer (MAC). edges: {bank: active up + down connections}; routes: bank ids
    per route, every route sharing bank 0 (MoE) or one route of one bank (full). Mean over routes."""
    return sum(sum(edges[b] for b in route) for route in routes)/len(routes)


def worth_aligning(work):
    return work >= MIN_WORK_PER_TOKEN


def capacity(count):
    """Capacity the FFN reserves for a work buffer of `count` floats."""
    return 1 << (count-1).bit_length()


def same_capacity(pairs):
    """pairs: (padded floats, published floats) per work buffer."""
    return all(capacity(padded) == capacity(published) for padded, published in pairs)


# -- one segment, one bank (full model) ---------------------------------------------------------------------
def one_segment_source(kind, text):
    """The published one-segment source with the row stride rounded up; nothing else is rewritten."""
    new = ONE_SEGMENT_SITE.replace('(ns[0])', aligned_expression('ns[0]'))
    if text.count(ONE_SEGMENT_SITE) != ONE_SEGMENT_SITES[kind] or new in text:
        raise RuntimeError(f'unexpected published {kind} source')
    return text.replace(ONE_SEGMENT_SITE, new)


def one_segment_aligned(n, hidden, d_model):
    """True when a one-segment call of n tokens takes the aligned layout."""
    if n <= 4 or n % ALIGN == 0:
        return False
    stride = aligned(n)
    return same_capacity(((d_model*stride, d_model*n), (hidden*stride, hidden*n)))


# -- bank spans (MoE) ---------------------------------------------------------------------------------------
def stride_layout(segments, stride):
    """`span_layout` with each span's row stride, and buffer offsets accumulated on those strides."""
    rows, offsets = [], {0: '0u', 1: '0u'}
    for span in K.span_layout(segments):
        s = stride(span['n'])
        rows.append(dict(span, stride=s, offset=offsets[span['buffer']]))
        offsets[span['buffer']] = f"({offsets[span['buffer']]}+{span['width']}u*{s})"
    return rows


def up_source(segments, stride=aligned_expression):
    """`up_source_spans_runtime_widths` with the row strides of X and A given by `stride`."""
    args = ['constant uint* ns', 'constant uint* bw']
    for b in K._banks(segments):
        args += [f'device const int* o{b}', f'device const short* i{b}', f'device const float* w{b}']
    args += ['device const float* X']+[f'device float* A{k}' for k in sorted({s['buffer'] for s in K.span_layout(segments)})]
    assert len(args) <= K.MAX_ARGS
    row = stride(K._sum(0, len(segments)))
    code, start = 'uint lane=tid&31u,simd=tid>>5u;', '0u'
    offsets = {0: '0u', 1: '0u'}
    for span in K.span_layout(segments):
        b, n, k = span['bank'], span['n'], span['buffer']
        h = f'bw[{b}]'; count = f'(({h}+15u)/16u)'
        total = f'({count}*({n}<64u?1u:(({n}+127u)/128u)))'
        head = (f'uint z=tg-{start},tile=z/{count},r=(z%{count})*16u+simd;if(r>={h})return;'
                f'uint begin=uint(o{b}[r]),end=uint(o{b}[r+1]);')
        x = lambda i, t, o=span['origin']: f'X[{i}*{row}+{o}+{t}]'
        store = lambda r, t, off=offsets[k], k=k, s=stride(n): f'A{k}[{off}+{r}*{s}+{t}]'
        body = f'if({n}<64u){{{K._up_body(b, h, n, 1, x, store)}}}else{{{K._up_body(b, h, n, 2, x, store)}}}'
        code += f'if(tg>={start} && tg<({start}+{total})){{{head}{body}}}'
        start = f'({start}+{total})'
        offsets[k] = f'({offsets[k]}+{h}*{stride(n)})'
    return K.PRELUDE+f'kernel void up({",".join(args)},{K.IDS}){{{code}}}'


def down_source(segments, modes, stride=aligned_expression, *, d_model):
    """`down_source_spans` with each span read at its `stride`."""
    layout = stride_layout(segments, stride)
    span_of = {(s['bank'], g): s for s in layout for g in range(s['first'], s['end'])}

    def act_of(g):
        def act(b, q, t):
            s = span_of[b, g]
            return f'A{s["buffer"]}[({s["offset"]})+uint(h{b}[{q}])*{s["stride"]}+{K._sum(s["first"], g)}+{t}]'
        return act
    buffers = sorted({s['buffer'] for s in layout})
    return K._down_kernel(segments, modes, None, lambda positions: [f'device const float* A{k}' for k in buffers], act_of,
                         d_model)


def span_buffers(segments, ns):
    """{buffer: floats} of the activation buffers with aligned row strides."""
    sizes = {}
    for span, n in zip(K.span_layout(segments), K.span_counts(segments, ns)):
        sizes[span['buffer']] = sizes.get(span['buffer'], 0)+span['width']*aligned(n)
    return dict(sorted(sizes.items()))


def span_metadata(segments, ns):
    """`span_metadata` on aligned strides: per segment and bank, (span offset, span row stride, token delta)."""
    offsets, spans = {0: 0, 1: 0}, {}
    for s, n in zip(K.span_layout(segments), K.span_counts(segments, ns)):
        spans[s['bank'], s['first']] = (offsets[s['buffer']], aligned(n), s['first'], s['end'])
        offsets[s['buffer']] += s['width']*aligned(n)
    out = []
    for g, (banks, _) in enumerate(segments):
        row = []
        for b in banks:
            offset, stride, first, _ = next(v for (bank, _), v in spans.items() if bank == b and v[2] <= g < v[3])
            row += [offset, stride, sum(ns[first:g])]
        out += row+[0]*(K.SPAN_STRIDE-len(row))
    return tuple(out)


def spans_aligned(segments, ns, d_model):
    """True when a bank-major call with bank spans takes the aligned layout."""
    if min(ns) <= 4 or all(n % ALIGN == 0 for n in ns):
        return False
    total = sum(ns)
    padded, published = span_buffers(segments, ns), K.span_buffers(segments, ns)
    return same_capacity([(d_model*aligned(total), d_model*total)]+[(padded[k], published[k]) for k in published])
