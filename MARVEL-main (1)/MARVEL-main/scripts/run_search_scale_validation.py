"""Predeclared bounded technical smoke; no final acceptance claim."""
from pathlib import Path
import sys,subprocess,json,os,hashlib
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'artifacts/search_recovery_independent_validation_20261010.json'
plan=json.loads(PLAN.read_text())
assert plan['smoke_only'] and plan['horizon_seconds']==80
results=[]
for count in (60,):
    for seed in plan['seeds']:
        relative=f'artifacts/search_independent_{count}_{seed}_20261010'
        folder=ROOT/relative
        command=[sys.executable,'-B',str(ROOT/'scripts/benchmark_fair_search.py'),
            '--out',relative,'--uavs',str(count),'--seed',str(seed),
            '--steps','80','--physical-coverage','--stop-on-completion']
        print(f'PREDECLARED_CASE uavs={count} seed={seed}',flush=True)
        completed=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1'})
        if completed.returncode:
            results.append({'uavs':count,'seed':seed,'invalid_run_exit':completed.returncode})
        else:
            manifest=json.loads((folder/'manifest.json').read_text())
            snapshot={}
            for name,expected in manifest['source_hashes'].items():
                content=(ROOT/name).read_bytes()
                assert hashlib.sha256(content).hexdigest()==expected,name
                if name.endswith(('.py','.yaml')):snapshot[name]=content.decode()
            (folder/'source_snapshot.json').write_text(json.dumps(snapshot,indent=2))
            paired=json.loads((folder/'paired.json').read_text())
            gppo,rule=paired
            gain=rule['metric']['mean_capped_seconds']/gppo['metric']['mean_capped_seconds']-1
            results.append({'uavs':count,'seed':seed,'paired':paired,'efficiency_gain':gain,
                'valid':all(item['valid_for_comparison'] for item in paired)})
            print(json.dumps({'CASE_RESULT':count,'seed':seed,'gain':gain,
                'gppo':gppo['metric'],'rule':rule['metric']},ensure_ascii=False),flush=True)
        (ROOT/'artifacts/search_independent_scale_validation_results_20261010.json').write_text(json.dumps(
            {'plan':plan,'results':results,'smoke_only':True,'final_project_acceptance':False},indent=2))
print('PREDECLARED_BATCH_FINISHED',flush=True)
