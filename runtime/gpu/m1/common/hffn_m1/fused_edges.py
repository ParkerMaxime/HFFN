"""Common owner-tile adjoint/outer-product source, with cold ABI lowering.

The caller owns traversal, row/slot decoding and load placement. Preserve
native constant-degree loops and preloaded activations rather than hiding
them behind dynamic endpoints or pointer aliases in a Metal helper.
"""
from pathlib import Path
import json,re

def edge_operation(direction, *, native=False, weight='weight'):
    if direction not in ('down','input'):
        raise ValueError('unknown fused edge direction')
    if native:
        weight='down' if direction=='down' else 'up'
    reduced='st' if native else 'reduced'
    intrinsic='simd_sum' if native else 'metal::simd_sum'
    base='F*D+' if native and direction=='down' else 'UN+' if direction=='down' else ''
    lines=[f'float w={weight}[p],dot=0.0f;']
    for k in range(4):
        value=f'gy[o*N+n{k}]' if direction=='down' else f'gz[r*N+n{k}]'
        left=(f'h{k}' if native else f'act[r*N+n{k}]') if direction=='down' else 'v'
        right='v' if direction=='down' else f'x[i*N+n{k}]'
        lines.append(f'    if(n{k}<N) {{ float v={value}; a{k}+=w*v; dot+={left}*{right}; }}')
    lines += [f'    float {reduced}={intrinsic}(dot);',
              f'    if(lane==0u) partial[({base}p)*OWNER_TILES+tile]={reduced};' if base else
              f'    if(lane==0u) partial[p*OWNER_TILES+tile]={reduced};']
    source='\n'.join(lines)
    if native:
        # Only lexical whitespace changes; the expressions are generated once.
        source=re.sub(r'(?<=\w)\s+(?=\w)','\x01',source)
        source=re.sub(r'\s+','',source).replace('\x01',' ')
    return source

def emit_header(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    lines=['#pragma once']
    for direction in ('down','input'):
        lines.append('#define HFFN_M1_FUSED_'+direction.upper()+'_OP '+json.dumps(edge_operation(direction,native=True)))
    path.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--emit-header',required=True)
    emit_header(parser.parse_args().emit_header)
