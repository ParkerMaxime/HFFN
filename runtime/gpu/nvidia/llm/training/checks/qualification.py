"""One bounded S2 functional campaign: contexts, 70M d10, then 410M d80.

Run on the CUDA VM through launch.sh qualify. No timing criterion is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

import context_check as helpers

HERE = Path(__file__).resolve().parent
STAGES = ("contexts", "70m_d10_n4", "410m_d80_n4")
BUDGET_SECONDS = 20 * 60


class CampaignTimeout(Exception):
    pass


def expire(_signum, _frame):
    raise CampaignTimeout("S2 work budget expired or session termination requested")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("qualification executes only on the CUDA VM")
    reserved = ("manifest.json", "config.json", "provenance.json", "summary.json", "raw.jsonl", "EXIT")
    if any((args.out / name).exists() for name in reserved):
        raise RuntimeError("existing campaign records cannot be overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"mode": "qualify", "planned_ids": list(STAGES), "work_budget_seconds": BUDGET_SECONDS,
              "attention_backend": "math", "batch": 1, "seq_len": 4, "updates": 5,
              "timing": "unmeasured; whole-model <=1.10 criterion remains open",
              "scope": "functional qualification only; no promotion verdict", "attempt": args.attempt}
    helpers.write_json(args.out / "config.json", config)
    source = helpers.provenance(args.base)
    helpers.write_json(args.out / "provenance.json", source)
    manifest = {"status": "running", "planned": len(STAGES), "done": 0,
                "planned_ids": list(STAGES), "done_ids": [], "full_coverage": False}
    helpers.update_manifest(args.out / "manifest.json", manifest)
    results, error, code = {}, None, 1
    model_token_digest = None
    signal.signal(signal.SIGALRM, expire)
    signal.signal(signal.SIGTERM, expire)
    signal.alarm(BUDGET_SECONDS)
    with (args.out / "raw.jsonl").open("x") as raw:
        def record(row):
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            raw.flush()
            print(json.dumps(row, allow_nan=False), flush=True)

        try:
            for stage in STAGES:
                child_dir = args.out / stage
                child_dir.mkdir()
                checker = "context_check.py" if stage == "contexts" else "model_check.py"
                command = [sys.executable, "-u", "-B", str(HERE / checker), "--out", str(child_dir), "--base", args.base]
                if stage != "contexts":
                    export = "pythia70m_sparse10_ffn_only" if stage.startswith("70m") else "pythia410m_sparse80_ffn_only"
                    command += ["--mode", "run", "--attention-backend", "math", "--export", export]
                record({"event": "child_start", "stage": stage, "command": command})
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, start_new_session=True)
                try:
                    with (child_dir / "run.log").open("x") as log:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            try:
                                row = json.loads(line)
                            except ValueError:
                                print(f"[{stage}] {line.rstrip()}", flush=True)
                            else:
                                progress = {key: row[key] for key in ("id", "event", "arm", "step", "phase", "pass", "error") if key in row}
                                print(f"[{stage}] {json.dumps(progress)}", flush=True)
                    child_code = process.wait()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                    process.stdout.close()
                if child_code != 0:
                    raise RuntimeError(f"{stage} process failed: {child_code}; no later stage runs")
                state = json.loads((child_dir / "manifest.json").read_text())
                if ((child_dir / "EXIT").read_text().strip() != "0" or state.get("status") != "passed"
                        or state.get("full_coverage") is not True):
                    raise RuntimeError(f"{stage} has no passed/full-coverage/EXIT 0 record")
                child_source = json.loads((child_dir / "provenance.json").read_text())
                if child_source.get("base") != source["base"] or any(
                    child_source.get("source_hashes", {}).get(path) != digest
                    for path, digest in source["source_hashes"].items()
                ):
                    raise RuntimeError(f"{stage} ran on different sources")
                if stage != "contexts":
                    child_config = json.loads((child_dir / "config.json").read_text())
                    if (child_config["attention_backend"], child_config["batch"], child_config["seq_len"],
                            child_config["steps"]) != ("math", 1, 4, 5):
                        raise RuntimeError(f"{stage} ran a different model protocol")
                    token_digest = child_source.get("token_ids_sha256")
                    if not isinstance(token_digest, str) or len(token_digest) != 64:
                        raise RuntimeError(f"{stage} has no token identity record")
                    if model_token_digest is not None and token_digest != model_token_digest:
                        raise RuntimeError("the two model stages used different token IDs")
                    model_token_digest = token_digest
                results[stage] = {"status": "passed", "exit_code": 0, "full_coverage": True,
                                  "run_path": stage, "checks": state.get("checks", {})}
                record({"id": stage, **results[stage]})
                manifest.update(done=len(results), done_ids=list(results), full_coverage=list(results) == list(STAGES))
                helpers.update_manifest(args.out / "manifest.json", manifest)
            for path, digest in source["source_hashes"].items():
                if hashlib.sha256((helpers.REPO / path).read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"source changed during the campaign: {path}")
            code = 0
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            code = 124 if isinstance(exc, CampaignTimeout) else 1
            record({"event": "failure", "error": error})
            traceback.print_exc()
        finally:
            signal.alarm(0)
    passed = code == 0 and list(results) == list(STAGES)
    manifest.update(status="passed" if passed else "failed", exit_code=code,
                    done=len(results), done_ids=list(results), full_coverage=list(results) == list(STAGES), error=error)
    helpers.update_manifest(args.out / "manifest.json", manifest)
    helpers.write_json(args.out / "summary.json", {"passed": passed, "results": results, "error": error,
                                                  "functional_only": True, "time_gate": "unmeasured; no promotion"})
    with (args.out / "EXIT").open("x") as output:
        output.write(f"{code}\n")
    print(f"S2 qualification: {manifest['status']} ({len(results)}/3), EXIT {code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
