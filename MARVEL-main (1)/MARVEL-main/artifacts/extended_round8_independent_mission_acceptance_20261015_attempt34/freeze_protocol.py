"""Freeze one candidate and unseen complete-mission acceptance before outcomes."""
from pathlib import Path
import json,hashlib,subprocess,datetime,os,shutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=read(ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33/plan.json')
for f,h in old['source_hashes'].items():assert sha(ROOT/f)==h,f
model=old['models']['round8'];assert sha(ROOT/model['path'])==model['sha256']
seeds=[101101,202202,303303];consumed=[];examined=0
# Only actual rollout seed declarations count; reservation lists do not.
for folder,dirs,files in os.walk(ROOT/'artifacts'):
 dirs[:]=[d for d in dirs if d not in ['source_snapshot','__pycache__','.pytest_cache']]
 if Path(folder)==OUT:dirs[:]=[];continue
 for name in files:
  if name not in ['manifest.json','run_started.json','complete.json','initial.json','config.json']:continue
  p=Path(folder)/name
  try:j=read(p)
  except (ValueError,OSError):continue
  if not isinstance(j,dict):continue
  examined+=1;values=[j.get('seed'),j.get('case',{}).get('seed'),j.get('args',{}).get('seed')]
  values+=j.get('args',{}).get('seeds',[])
  values+=[j.get('scenario',{}).get('random_seed'),j.get('task_scheduler',{}).get('search_scenario_seed')]
  if any(v in seeds for v in values):consumed.append(str(p.relative_to(ROOT)))
assert not consumed,consumed
plan={k:old[k] for k in ['head','branch','git_root','scenario','warning_rate_mps','horizon_seconds','wall_budget_seconds_per_arm','frozen_checkpoint','frozen_sha256','randomness','source_policy']}
plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models={'round8':model},arms=['rule','round8'],cases=[dict(seed=s,uavs=n,dynamic=d) for s in seeds for n in [30,60] for d in [False,True]],workers=4,training_permitted=False,optimizer_updates=0,deterministic=True,python_hash_seed=0,reserved_seeds_untouched=[],reserved_final_seeds=seeds,independent_seed_audit=dict(prior_json_records_examined=examined,prior_actual_consumptions=consumed),metric=dict(primary='1 - sum(candidate completed joint mission seconds)/sum(rule completed joint mission seconds)',threshold=.15,aggregation='12 equally weighted paired cases; no dropping failures or selecting groups',completion='All nonrelay tasks complete AND actual radio>=0.9 AND Safety fully released, continuously for3s; full300s physical window must remain safe',failure='Any candidate or baseline incomplete/unsafe case fails acceptance eligibility; report censored300s separately, never exclude it',secondary='sum(rule time)/sum(candidate time)-1, Search sensor Find latency; neither replaces primary',groups='Report30-static,30-dynamic,60-static,60-dynamic and per-seed results; no subgroup regression required',confidence='Seed-cluster bootstrap percentile95%, descriptive only with3 independent seeds; point-estimate gate is predeclared, not proof lower confidence bound>=15%'),pass_rule='All24 full H300 runs valid, all12 pairs jointly complete and safe; pooled primary>=0.15; each four domain-group primary>=0.0',scope='Unseen seeds on unchanged connected80m scenario. Does not establish original500m deployment/map generalization.',no_adaptation='One frozen round8 only. Do not tune/retrain/select another checkpoint after viewing reserved outcomes. Preserve partial attempts; infrastructure retry only explicitly labelled.',git_status_before=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==plan['head']
assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==plan['branch']
plan['source_hashes']=old['source_hashes'].copy()
for p in [OUT/'evaluate_arm.py',OUT/'run_acceptance.py',OUT/'freeze_protocol.py']:
 plan['source_hashes'][str(p.relative_to(ROOT))]=sha(p)
for f in plan['source_hashes']:
 p=ROOT/f
 if p.suffix in ['.py','.yaml','.png']:
  dest=OUT/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
(OUT/'plan.json').write_text(json.dumps(plan,indent=2))
(OUT/'protocol_sha256.txt').write_text(sha(OUT/'plan.json')+chr(10))
print(json.dumps(dict(status='FROZEN_BEFORE_RUNS',out=str(OUT),cases=12,runs=24,seed_audit=plan['independent_seed_audit'],metric=plan['metric'],pass_rule=plan['pass_rule']),ensure_ascii=False),flush=True)
