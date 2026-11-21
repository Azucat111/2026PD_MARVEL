"""Fresh independent verification of completed frozen acceptance outputs."""
from pathlib import Path
import json,hashlib,sys,subprocess,datetime
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(OUT/'plan.json');r=read(OUT/'acceptance_complete.json')
assert r['status']=='COMPLETE' and r['runs']==24 and len(r['rows'])==12
assert sha(OUT/'plan.json')==(OUT/'protocol_sha256.txt').read_text().strip()==r['protocol_sha256']
for f,h in p['source_hashes'].items():assert sha(ROOT/f)==h,f
for m in p['models'].values():assert sha(ROOT/m['path'])==m['sha256']
assert sha(Path(p['frozen_checkpoint']))==p['frozen_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==p['head']
sys.path.insert(0,str(ROOT/'scripts'))
from gppo_joint_completion_metric import joint_completion_metric
records=[]
for i,c in enumerate(p['cases']):
 for a in p['arms']:
  base=OUT/'data'/f"case_{i:02d}_{c['seed']}_{c['uavs']}_{'dynamic' if c['dynamic'] else 'static'}"/a
  complete=read(base/'complete.json');folder=base/('rule' if a=='rule' else 'gppo')
  assert complete['case']==c and complete['source_stable'] and complete['optimizer_updates']==0
  trace=read(folder/'physical_joint_trace.json');assert len(trace)==3000
  recomputed=joint_completion_metric(trace,300,physics_dt=.1)
  assert recomputed==complete['summary']['secondary_joint_metric']==r['rows'][i]['arms'][a]['joint']
  records.append(dict(case=c,arm=a,full_physics_samples=len(trace),metric=recomputed,trace_sha256=sha(folder/'physical_joint_trace.json')))
b=sum(x['metric']['secondary_joint_completion_seconds'] for x in records if x['arm']=='rule')
c=sum(x['metric']['secondary_joint_completion_seconds'] for x in records if x['arm']=='round8')
assert abs((1-c/b)-r['primary']['completed_time_reduction'])<1e-12
expected=r['primary']['all_pairs_eligible'] and 1-c/b>=.15 and all(g['completed_time_reduction']>=0 for g in r['groups'].values())
assert r['verdict']==('PASS' if expected else 'FAIL')
tracked=subprocess.check_output(['git','status','--porcelain','-uno'],cwd=ROOT,text=True)
before='\n'.join(x for x in p['git_status_before'].splitlines() if not x.startswith('??'))
assert before.strip()==tracked.strip(),'Tracked workspace status changed'
v=dict(status='VERIFIED',verdict=r['verdict'],runs=24,physical_samples=72000,primary=r['primary'],groups=r['groups'],source_and_weights_unchanged=True,tracked_workspace_unchanged=True,optimizer_updates=0,protocol_sha256=r['protocol_sha256'],verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records)
(OUT/'verification_complete.json').write_text(json.dumps(v,indent=2))
print(json.dumps({k:x for k,x in v.items() if k!='records'}),flush=True)
