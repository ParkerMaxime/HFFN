"""Cold CLI adapter; arithmetic/source generation stays inference-owned."""
import os
import subprocess
from pathlib import Path
import sys
from native_context import prepare_sources

def main():
    arguments=sys.argv[1:]
    payload=None
    if '--export-dir' in arguments:
        from export_input import read_export,checked
        at=arguments.index('--export-dir')
        if at+1>=len(arguments):raise ValueError('--export-dir requires a path')
        payload=read_export(arguments[at+1]);del arguments[at:at+2]
        i,h,o,_,_=checked(payload)
        arguments+=['--input',str(i),'--hidden',str(h),'--output-size',str(o),'--degree','1']
    directory=prepare_sources(arguments)
    if '--hffn-prepare-only' in arguments:
        print(directory,flush=True)
        return
    prototype='--hffn-forward-prototype' in arguments
    name='metal_hffn_training_forward_runner.native' if prototype else 'metal_hffn_train_step_runner.native'
    leaf=Path(__file__).resolve().parent
    binary=leaf/'build'/name
    if prototype and not binary.is_file():
        root=next(p for p in leaf.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
        binary=root/'runtime/gpu/m1/ffn/training/build'/name
    os.environ['HFFN_INFERENCE_FORWARD_DIR']=str(directory)
    command=[str(binary),*(a for a in arguments if a!='--hffn-forward-prototype')]
    if payload is None:os.execv(str(binary),command)
    if prototype:raise ValueError('export support is not admitted by the ReLU prototype')
    from export_input import parts
    child=subprocess.Popen([*command,'--support-stdin'],stdin=subprocess.PIPE)
    try:
        for part in parts(payload):child.stdin.write(part)
        child.stdin.close()
    except BrokenPipeError:pass
    raise SystemExit(child.wait())
if __name__=='__main__':main()
