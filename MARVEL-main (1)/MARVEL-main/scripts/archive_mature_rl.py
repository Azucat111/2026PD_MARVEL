"""Copy scored and censored physical evidence into a reproducible Git archive."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def archive(live, target):
    runs = target / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    files = ("start.txt", "historical-build.log", "expanded-build.log", "study.jsonl", "freeze.json",
             "benchmark.json", "experiments.csv", "asset-verification.txt", "regression-tests.log",
             "inherited-regression.json", "inherited-regression.log", "checkpoint-parity.log")
    suffixes = (".train.jsonl", ".pth", "-replay.json.gz", "-audit.json", "-fit.json")
    for source in sorted(live.iterdir()):
        if source.is_file() and (source.name in files or source.name.endswith(suffixes)):
            shutil.copy2(source,target/source.name)
        if not source.is_dir():
            continue
        summaries = [p for p in (source/"summary.json",source/"R0.json",source/"R1.json") if p.is_file()]
        if not summaries:
            continue
        dest = runs / source.name
        dest.mkdir(exist_ok=True)
        for path in summaries:
            data = path.read_bytes()
            json.loads(data)
            (dest / f"{path.name}.gz").write_bytes(gzip.compress(data,mtime=0))
        for name in ("events.jsonl.gz","public.zip","paired.json","learning_policy.json"):
            path = source/name
            if path.is_file():
                shutil.copy2(path,dest/name)
        log = source.with_suffix(".log")
        if log.is_file():
            shutil.copy2(log,dest/"run.log")
    manifest = target / "SHA256SUMS"
    with manifest.open("w") as handle:
        for path in sorted(target.rglob("*")):
            if path.is_file() and path != manifest:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                print(f"{digest}  {path.relative_to(target).as_posix()}",file=handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",type=Path,required=True)
    parser.add_argument("--target",type=Path,required=True)
    args = parser.parse_args()
    archive(args.live,args.target)


if __name__=="__main__":
    main()
