"""Step launcher: `python -m lib.campaign --step-dir <step> --mode check|smoke|run [--resume <run dir>]`.

Reads `<step>/step.json`, runs the checks, then the stages (jobs of a stage in parallel, one process per job), and
writes the run folder `<step>/runs/<mode>_<UTC>/` with run.log, manifest.json (status of every job), checks.json,
one folder per job (config.json, provenance.json, raw.jsonl, metrics.csv, summary.json, run.log, EXIT) and EXIT.
Progress is printed unbuffered. Exit code 0 only if every job exited 0.
"""
import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
STAGE_KEYS = ("impl", "schedule", "horizon", "tokens_per_step", "save_at", "valid_bytes", "warmup", "mixing",
              "test", "final_windows", "eval_every", "valid_windows")


class Log:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg):
        line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def utc():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def run_checks(run_dir, log, scripts=("check_model.py",)):
    ok = True
    for script in scripts:
        stem = Path(script).stem
        out = run_dir / f"{stem}.json" if len(scripts) > 1 else run_dir / "checks.json"
        with open(run_dir / f"{stem}.log", "w") as f:
            rc = subprocess.call([sys.executable, "-u", str(TRAIN / "checks" / script), str(out)],
                                 cwd=TRAIN, stdout=f, stderr=subprocess.STDOUT)
        res = json.loads(out.read_text()) if out.exists() else {"checks": [], "failed": ["crashed"]}
        log(f"checks {script}: {len(res['checks']) - len(res['failed'])}/{len(res['checks'])} passed (exit {rc})")
        ok &= rc == 0
    return ok


def resolve_lr(spec, arm, selection, smoke):
    v = spec[arm] if isinstance(spec, dict) else spec
    if isinstance(v, str) and v.startswith("sweep:"):
        if smoke:
            return 2e-3
        return selection["best"][v.split(":", 1)[1]]
    return float(v)


def job_cmd(job, threads):
    c = ["-u", "-m", "lib.train", "--arm", job["arm"], "--seed", str(job["seed"]), "--steps", str(job["steps"]),
         "--lr", repr(job["lr"]), "--out", str(job["dir"]), "--threads", str(threads),
         "--eval-every", str(job.get("eval_every", 500)), "--valid-windows", str(job.get("valid_windows", 4096))]
    if job.get("mixing", True):
        c += ["--mixing"]
    for key, flag in (("schedule", "--schedule"), ("horizon", "--horizon"), ("tokens_per_step", "--tokens-per-step"),
                      ("save_at", "--save-at"), ("resume", "--resume"), ("valid_bytes", "--valid-bytes"),
                      ("warmup", "--warmup")):
        if job.get(key) is not None:
            c += [flag, str(job[key])]
    if job.get("final_windows"):
        c += ["--final-windows", str(job["final_windows"])]
    if job.get("impl"):
        c += ["--impl", job["impl"]]
    if job.get("overrides"):
        c += ["--arm-overrides", json.dumps(job["overrides"])]
    if job.get("test"):
        c += ["--test"]
    return [sys.executable, *c]


def run_jobs(jobs, run_dir, workers, threads, manifest, log, resume):
    def one(job):
        d = Path(job["dir"])
        d.mkdir(parents=True, exist_ok=True)
        key = str(d.relative_to(run_dir))
        ex = d / "EXIT"
        if resume and ex.exists() and ex.read_text().strip() == "0":
            manifest["jobs"][key] = {"status": "passed", "exit": 0, "resumed": True}
            log(f"skip (already passed) {key}")
            return key, 0
        manifest["jobs"][key] = {"status": "running", "started": utc()}
        write_manifest(run_dir, manifest)
        log(f"start {key}")
        with open(d / "run.log", "w") as f:
            rc = subprocess.call(job_cmd(job, threads), cwd=TRAIN, stdout=f, stderr=subprocess.STDOUT)
        if not ex.exists():  # crashed before writing its own EXIT: reliable exit code
            ex.write_text(f"{rc or 1}\n")
        summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        manifest["jobs"][key] = {"status": summary.get("status", "crashed"), "exit": rc, "ended": utc(),
                                 "final_valid_bpb": summary.get("final_valid_bpb"), "cause": summary.get("cause")}
        write_manifest(run_dir, manifest)
        log(f"done  {key}: exit {rc}, valid bpb {summary.get('final_valid_bpb')}")
        return key, rc

    # longest jobs first (more layers, more steps) so that the pool is busy until the end
    from .arms import get_arm

    def weight(j):
        a = get_arm(j["arm"])
        return j["steps"] * a.n_layers * (1.7 if a.context > 128 else 1.0)

    jobs = sorted(jobs, key=lambda j: -weight(j))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return [r for r in ex.map(one, jobs)]


def write_manifest(run_dir, manifest):
    with LOCK:
        tmp = run_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(run_dir / "manifest.json")


def select_lr(stage_dir, stage, log):
    best, table = {}, {}
    for arm in stage["arms"]:
        rows = []
        for lr in stage["lrs"]:
            for seed in stage["seeds"]:
                s = json.loads((stage_dir / f"{arm}_lr{lr:g}_s{seed}" / "summary.json").read_text())
                rows.append((s["final_valid_bpb"], lr))
        table[arm] = {f"{lr:g}": v for v, lr in rows}
        best[arm] = min(rows)[1]
    sel = {"best": best, "valid_bpb": table, "coincide": len(set(best.values())) == 1, "grid": stage["lrs"]}
    (stage_dir / "selection.json").write_text(json.dumps(sel, indent=1))
    log(f"learning-rate selection: {best} (coincide: {sel['coincide']})")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--mode", required=True, choices=["check", "smoke", "run"])
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    step_dir = Path(a.step_dir).resolve()
    spec = json.loads((step_dir / "step.json").read_text())
    run_dir = Path(a.resume).resolve() if a.resume else step_dir / "runs" / f"{a.mode}_{utc()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Log(run_dir / "run.log")
    log(f"step {spec['step']} mode {a.mode} -> {run_dir}")
    manifest = {"step": spec["step"], "mode": a.mode, "started": utc(), "jobs": {}}
    write_manifest(run_dir, manifest)

    def finish(code, why=""):
        manifest["ended"], manifest["exit"], manifest["why"] = utc(), code, why
        write_manifest(run_dir, manifest)
        (run_dir / "EXIT").write_text(f"{code}\n")
        log(f"EXIT {code} {why}")
        return code

    if not run_checks(run_dir, log, tuple(spec.get("checks", ["check_model.py"]))):
        return finish(3, "pre-training checks failed (harness)")
    if a.mode == "check":
        return finish(0)

    selection = None
    workers, threads = spec.get("workers", 4), spec.get("threads", 1)
    smoke = a.mode == "smoke" or (a.mode == "run" and spec.get("run_is_smoke"))
    rcs = []
    if smoke:
        seen, jobs = set(), []
        for st in spec["stages"]:
            if st.get("kind") == "branches":
                continue
            for arm in st["arms"]:
                if arm in seen:
                    continue
                seen.add(arm)
                jobs.append({"arm": arm, "seed": 0, "steps": 40, "lr": 2e-3, "dir": run_dir / "smoke" / f"{arm}_s0",
                             "eval_every": 20, "valid_windows": 256, "final_windows": 512,
                             **{k: st[k] for k in STAGE_KEYS if k in st}, **spec.get("smoke", {})})
        rcs += [rc for _, rc in run_jobs(jobs, run_dir, workers, threads, manifest, log, False)]
        pair = spec.get("smoke_branch")
        if pair:  # one plateau-and-branch pair
            plateau = {"arm": pair["arm"], "seed": 0, "steps": pair["plateau_steps"], "lr": pair["lr"],
                       "dir": run_dir / "smoke_branch" / "plateau", "eval_every": 1000, "valid_windows": 256,
                       "final_windows": 512, "schedule": "wsd", "horizon": pair["horizon"], "warmup": pair["warmup"],
                       "tokens_per_step": 8192, "save_at": str(pair["plateau_steps"]), "mixing": False}
            branch = dict(plateau, steps=pair["horizon"], dir=run_dir / "smoke_branch" / "branch", save_at=None,
                          resume=str(run_dir / "smoke_branch" / "plateau" / f"ckpt_{pair['plateau_steps']}.pt"))
            rcs += [rc for _, rc in run_jobs([plateau], run_dir, 1, threads, manifest, log, False)]
            rcs += [rc for _, rc in run_jobs([branch], run_dir, 1, threads, manifest, log, False)]
        return finish(0 if all(rc == 0 for rc in rcs) else 1)

    for st in spec["stages"]:
        stage_dir = run_dir / st["name"]
        jobs = []
        common = {k: st[k] for k in STAGE_KEYS if k in st}
        if st.get("kind") == "sweep":
            for arm in st["arms"]:
                for lr in st["lrs"]:
                    for seed in st["seeds"]:
                        jobs.append({"arm": arm, "seed": seed, "steps": st["steps"], "lr": float(lr),
                                     "dir": stage_dir / f"{arm}_lr{lr:g}_s{seed}", **common})
        elif st.get("kind") == "branches":
            for arm in st["arms"]:
                lr = resolve_lr(st["lr"], arm, selection, False)
                for seed in st["seeds"]:
                    for br in st["branches"]:
                        ck = run_dir / st["from"] / f"{arm}_s{seed}" / f"ckpt_{br['from_step']}.pt"
                        jobs.append({"arm": arm, "seed": seed, "steps": br["horizon"], "horizon": br["horizon"], "lr": lr,
                                     "resume": str(ck), "dir": stage_dir / f"{arm}_s{seed}_H{br['horizon']}",
                                     **{k: v for k, v in common.items() if k not in ("horizon", "save_at")}})
        else:
            for arm in st["arms"]:
                lr = resolve_lr(st["lr"], arm, selection, False)
                for seed in st["seeds"]:
                    jobs.append({"arm": arm, "seed": seed, "steps": st["steps"], "lr": lr,
                                 "dir": stage_dir / f"{arm}_s{seed}", **common,
                                 **({"overrides": st["overrides"][arm]} if arm in st.get("overrides", {}) else {})})
        log(f"stage {st['name']}: {len(jobs)} jobs, {workers} workers x {threads} thread(s)")
        rc = [r for _, r in run_jobs(jobs, run_dir, workers, threads, manifest, log, bool(a.resume))]
        rcs += rc
        if any(r != 0 for r in rc):
            return finish(1, f"stage {st['name']} has failed jobs")
        if st.get("kind") == "sweep":
            selection = select_lr(stage_dir, st, log)
    return finish(0 if all(r == 0 for r in rcs) else 1)


if __name__ == "__main__":
    sys.exit(main())
