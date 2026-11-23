"""Run the fixed-seed, bounded physical study in one four-worker queue."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts"
P2 = ROOT / "artifacts/joint_autoresearch_20261122/runs/models/p2-o-calibrated.json"
P2_SHA = "52b3424b64f0736641e1110eabbfeb83e43427cc549bdf191ea25d128a682276"
DOMAINS = ("30_static", "30_dynamic", "60_static", "60_dynamic")
TRAIN = tuple(f"{domain}_{24110601+i}" for i,domain in enumerate(DOMAINS))
EXTRA_TRAIN = tuple(f"{domain}_{24110611+i}" for i,domain in enumerate(DOMAINS))
ARCHIVE_DEV = ("30_static_1280021", "30_dynamic_1280022", "60_static_9164", "60_dynamic_1981204")
NEW_DEV = ("30_static_24110701", "60_dynamic_24110704")
VALIDATION = tuple(f"{domain}_{24110801+i}" for i,domain in enumerate(DOMAINS))
ALGORITHMS = ("ddqn", "sac")


def checkpoint_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_status(path, arm=None):
    filename = f"{arm}.json" if arm in ("R0", "R1") else "summary.json"
    try:
        return json.loads((path / filename).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def command(out, case, arm, wall, *, model=None, mixed_only=False):
    uavs, domain, seed = case.split("_")
    if arm in ("R0", "R1"):
        return ["joint_four_arm_eval.py","--uavs",uavs,"--domain",domain,"--seed",seed,
                "--arms",arm,"--wall-seconds",str(wall),"--out",str(out)]
    if arm in ("J0", "P2-O"):
        args = ["--policy","learned","--checkpoint",str(P2),"--sha256",P2_SHA] if arm == "P2-O" else ["--policy","J0"]
        return ["run_joint_autoresearch.py","--case",case,"--out",str(out),"--wall",str(wall),*args]
    args = []
    if model:
        args.extend(("--checkpoint",str(model),"--sha256",checkpoint_sha(model)))
    if arm == "explore":
        args.extend(("--behavior-seed",str(int(seed)+4821)))
        if mixed_only:
            args.append("--first-mixed-only")
    return ["run_mature_rl.py","--case",case,"--out",str(out),"--wall",str(wall),"--policy",arm,*args]


class Study:
    def __init__(self, live, start, workers):
        self.live, self.start, self.workers = live, start, workers
        self.stop_launch = start + 450*60
        self.stop_runs = start + 463*60

    def report(self, **row):
        row["time_utc"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(row, sort_keys=True), flush=True)

    def run_small(self, log_name, args):
        out = self.live / log_name
        call = [sys.executable, "-B", str(SCRIPT / args[0]), *args[1:]]
        self.report(job=log_name, action="start", command=call)
        with out.open("x") as handle:
            subprocess.run(call, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        self.report(job=log_name, action="done")

    def physical(self, jobs, *, attaching=(), phase_deadline=None):
        deadline = min(self.stop_runs, phase_deadline or self.stop_runs)
        pending = deque(jobs)
        external = set(attaching)
        active = {}
        while pending or external or active:
            for name, (proc, handle, arm) in list(active.items()):
                exit_code = proc.poll()
                if exit_code is None:
                    continue
                handle.close()
                data = read_status(self.live / name, arm) or {}
                self.report(job=name, action="done", exit_code=exit_code,
                            status=data.get("status"), score=data.get("score"))
                del active[name]
            for name in sorted(external):
                data = read_status(self.live / name)
                if data is not None or time.time() > self.start+4600:
                    self.report(job=name, action="attached_done", status=(data or {}).get("status"), score=(data or {}).get("score"))
                    external.remove(name)
            if pending and time.time() >= min(self.stop_launch, deadline - 1000):
                for job in pending:
                    self.report(job=job[0], action="skipped_budget")
                pending.clear()
            while pending and len(active)+len(external) < self.workers:
                name, case, arm, model, mixed_only = pending.popleft()
                out = self.live / name
                if read_status(out, arm) is not None:
                    self.report(job=name, action="reused")
                    continue
                wall = min(4500., deadline-time.time()-20.)
                args = command(out, case, arm, wall, model=model, mixed_only=mixed_only)
                if name.startswith("smoke-"):
                    args.extend(("--ticks", "100"))
                call = [sys.executable,"-B",str(SCRIPT / args[0]),*args[1:]]
                handle = (self.live / f"{name}.log").open("x")
                proc = subprocess.Popen(call, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                active[name] = (proc, handle, arm)
                self.report(job=name, action="start", command=call, pid=proc.pid)
            if pending or external or active:
                time.sleep(10)

    def collect(self):
        names = [f"j0-{case}" for case in TRAIN]
        jobs = [(f"explore-{case}", case, "explore", None, True) for case in TRAIN]
        jobs += [(f"explore-{case}", case, "explore", None, False) for case in EXTRA_TRAIN]
        smoke = [(f"smoke-archive-{algo}", ARCHIVE_DEV[0], algo, self.live/f"archive-{algo}.pth", False)
                 for algo in ALGORITHMS if (self.live/f"archive-{algo}.pth").is_file()]
        self.physical(smoke+jobs, attaching=names, phase_deadline=self.start+210*60)
        dirs = [self.live / name for name in names+[j[0] for j in jobs]]
        dirs = sorted(p for p in dirs if read_status(p))
        self.run_small("expanded-build.log",
                       ["build_mature_rl_replay.py","--parent-replay",str(self.live/"historical-replay.json.gz"),
                        "--runs",*map(str,dirs),
                        "--out",str(self.live/"expanded-replay.json.gz"),"--audit",str(self.live/"expanded-audit.json")])
        for algo in ALGORITHMS:
            self.run_small(f"expanded-{algo}.train.jsonl",
                           ["train_mature_rl.py","--replay",str(self.live/"expanded-replay.json.gz"),
                            "--algo",algo,"--seed","6177","--updates","600",
                            "--out",str(self.live/f"expanded-{algo}.pth")])

    def development(self):
        smoke = [(f"smoke-{algo}", ARCHIVE_DEV[0], algo, self.live/f"expanded-{algo}.pth", False)
                 for algo in ALGORITHMS]
        self.physical(smoke, phase_deadline=self.start+350*60)
        jobs = [(f"dev-expanded-{algo}-{case}", case, algo, self.live/f"expanded-{algo}.pth", False)
                for case in NEW_DEV+ARCHIVE_DEV for algo in ALGORITHMS]
        jobs += [(f"dev-archive-{algo}-{case}", case, algo, self.live/f"archive-{algo}.pth", False)
                 for case in NEW_DEV for algo in ALGORITHMS if (self.live/f"archive-{algo}.pth").is_file()]
        self.physical(jobs, phase_deadline=self.start+350*60)
        options = []
        for algo in ALGORITHMS:
            scores = [(read_status(self.live/f"dev-expanded-{algo}-{case}") or {}).get("score") for case in NEW_DEV]
            if all(s and s["joint_success"] and s["whole_window_safe"] for s in scores):
                options.append((sum(s["secondary_joint_completion_seconds"] for s in scores), algo))
        if not options:
            self.report(action="no_eligible_candidate")
            return None
        value, winner = min(options)
        model = self.live/f"expanded-{winner}.pth"
        freeze = dict(chosen=winner, model=model.name, checkpoint_sha256=checkpoint_sha(model),
                      new_development_cases=NEW_DEV, selection_scores=sorted(options),
                      chosen_development_sum_seconds=value,
                      frozen_utc=datetime.now(timezone.utc).isoformat(), validation_cases=VALIDATION)
        with (self.live/"freeze.json").open("x") as handle:
            json.dump(freeze, handle, indent=2, sort_keys=True)
        self.report(action="frozen", **freeze)
        return freeze

    def validation(self, freeze):
        model = self.live/freeze["model"]
        if checkpoint_sha(model) != freeze["checkpoint_sha256"]:
            raise ValueError("Validation checkpoint differs from the freeze")
        name = freeze["chosen"]
        jobs = [(f"validation-{arm}-{case}", case, arm, model if name == arm else None, False)
                for case in VALIDATION for arm in (name,"R1","P2-O","R0")]
        self.physical(jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phase", choices=("all","collect","dev","validation"), default="all")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Grant each physics worker one of up to four cores")
    start = datetime.fromisoformat((args.live/"start.txt").read_text().strip()).timestamp()
    controller = Study(args.live, start, args.workers)
    if args.phase in ("all","collect"):
        controller.collect()
    if args.phase in ("all","dev"):
        freeze = controller.development()
    elif args.phase == "validation":
        freeze = json.loads((args.live/"freeze.json").read_text())
    else:
        freeze = None
    if args.phase in ("all","validation") and freeze:
        controller.validation(freeze)


if __name__ == "__main__":
    main()
