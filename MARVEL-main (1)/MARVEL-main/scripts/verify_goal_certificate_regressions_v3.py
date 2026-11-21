from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[1]
files=sorted(str(p.relative_to(root)) for p in (root/'platform_tests').glob('test_extended_*.py'))
files+=['platform_tests/'+x for x in ['test_marvel_native_motion_parity.py','test_marvel_native_parity.py','test_marvel_native_policy_parity.py','test_marvel_native_sensing_parity.py','test_marvel_native_hazard_parity.py','test_search_route_support.py','test_observed_search_reachability.py','test_search_deadline_bound.py','test_common_benchmark_wall_budget.py','test_joint_completion_metric.py']]
result=subprocess.run([sys.executable,'-B','-m','pytest',*files,'-q'],cwd=root,text=True,capture_output=True)
print(result.stdout);print(result.stderr)
with (root/'artifacts/extended_goal_route_certificate_20261010_attempt1/regressions_v3.json').open('x') as stream:json.dump(dict(files=files,exit_code=result.returncode,output=result.stdout,passed=result.returncode==0,acceptance_claim=False),stream,indent=2)
raise SystemExit(result.returncode)
