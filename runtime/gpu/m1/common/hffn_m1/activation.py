"""One Metal activation definition; output formatting preserves existing ABIs."""
from pathlib import Path
import json
import re

_SOURCE=Path(__file__).with_suffix('.metal').read_text()
def _section(name):
    begin='// BEGIN '+name+'\n';end='\n// END '+name+'\n'
    if _SOURCE.count(begin)!=1 or _SOURCE.count(end)!=1:raise ValueError('invalid activation section: '+name)
    return _SOURCE.split(begin,1)[1].split(end,1)[0]
_ERF,_GELU,_GRAD=(_section(name) for name in ('erf','gelu','gradient'))
def inference_source():
    return '\n'+_ERF+'\n'+_GELU+'\n'
def training_source():
    return _ERF+'\n'+_GRAD
def native_source():
    # Formatting only: explicit metal namespace -> native's using namespace.
    def compact(s):
        s=re.sub(r'\s+',' ',s.replace('metal::',''))
        return re.sub(r'\s*([{}();,=+*/?:<>-])\s*',r'\1',s)
    return ''.join(compact(s)+'\n' for s in (_ERF,_GELU,_GRAD))
def emit_header(path):
    text='#pragma once\n#define HFFN_M1_ACTIVATION_MSL '+json.dumps(native_source())+'\n'
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(text)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--emit-header',required=True)
    emit_header(parser.parse_args().emit_header)
