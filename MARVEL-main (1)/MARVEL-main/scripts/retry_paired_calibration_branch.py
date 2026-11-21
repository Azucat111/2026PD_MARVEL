from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20'
plan=json.loads((OUT/'recovery_plan_20261015.json').read_text())
for f,h in plan['new_source_hashes'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
source=(ROOT/'scripts/collect_paired_calibration_branch.py').read_text();old="out=ROOT/f'artifacts/extended_paired_calibration_20261014_attempt20/data/seed_{a.seed}_rank_{a.rank}'"
assert source.count(old)==1
source=source.replace(old,"out=ROOT/f'artifacts/extended_paired_calibration_20261014_attempt20/data/seed_{a.seed}_rank_{a.rank}_retry_01'").replace('wall_budget_seconds=240','wall_budget_seconds=900')
exec(compile(source,str(ROOT/'scripts/collect_paired_calibration_branch.py'),'exec'),dict(__name__='__main__',__file__=str(ROOT/'scripts/collect_paired_calibration_branch.py')))
