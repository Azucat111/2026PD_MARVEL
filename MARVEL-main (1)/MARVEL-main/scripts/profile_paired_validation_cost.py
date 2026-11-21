"""One bounded existing-rule profile, not training or acceptance."""
from pathlib import Path
import sys,cProfile,pstats,io,json,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20/performance_probe'
OUT.mkdir(exist_ok=False)
source=(ROOT/'scripts/collect_paired_calibration_branch.py').read_text()
old="out=ROOT/f'artifacts/extended_paired_calibration_20261014_attempt20/data/seed_{a.seed}_rank_{a.rank}'"
assert source.count(old)==1
source=source.replace(old,"out=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20/performance_probe/physical_H6'")
assert source.count('a.seed,20,physical_coverage=True')==1
source=source.replace('a.seed,20,physical_coverage=True','a.seed,6,physical_coverage=True').replace('wall_budget_seconds=240','wall_budget_seconds=120').replace("summary['mission_steps']==20","summary['mission_steps']==6")
sys.argv=['profile_existing_rule','--seed','63410','--uavs','30','--rank','0','--dynamic']
profile=cProfile.Profile();start=time.monotonic();profile.enable()
try:exec(compile(source,str(ROOT/'scripts/collect_paired_calibration_branch.py'),'exec'),dict(__name__='__main__',__file__=str(ROOT/'scripts/collect_paired_calibration_branch.py')))
finally:
 profile.disable();profile.dump_stats(str(OUT/'profile.pstats'))
 stream=io.StringIO();pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(35);(OUT/'top_cumulative.txt').write_text(stream.getvalue())
 stream2=io.StringIO();pstats.Stats(profile,stream=stream2).strip_dirs().sort_stats('tottime').print_stats(25);(OUT/'top_self.txt').write_text(stream2.getvalue())
 print(stream.getvalue(),flush=True)
 (OUT/'profile_complete.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,training_updates=0,acceptance_claim=False),indent=2))
