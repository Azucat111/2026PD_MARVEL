from pathlib import Path
import json,hashlib,sys,subprocess,datetime
R=Path.cwd();O=Path(__file__).resolve().parent;D=O/'evaluation';OLD=R/'artifacts/gppo_three_hour_20261110';sys.path.insert(0,str(R/'scripts'))
from gppo_joint_completion_metric import joint_completion_metric
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
S=json.loads((OLD/'corrected_source_hashes.json').read_text());assert all(sha(R/f)==h for f,h in S.items());I=json.loads((OLD/'plan.json').read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert head==I['head'];assert all(sha(R/m['path'])==m['sha256'] for m in I['models'].values())
if not (D/'plan.json').exists():print('PLAN_NOT_READY');raise SystemExit()
P=json.loads((D/'plan.json').read_text());assert sha(D/'plan.json')==(D/'protocol_sha256.txt').read_text().strip();assert all(sha(R/f)==h for f,h in P['source_hashes'].items());assert all(sha(R/m['path'])==m['sha256'] for m in P['models'].values())
rows=[]
for i,c in enumerate(P['cases']):
 arms={};initial=[];hidden=[];configs=[]
 for a in P['arms']:
  folder=D/'data'/f"case_{i:02d}_{c['seed']}_{c['uavs']}_{'dynamic' if c['dynamic'] else 'static'}"/(a+'_original');f=folder/'complete.json'
  if not f.exists():continue
  x=json.loads(f.read_text());data=folder/('rule' if a=='rule' else 'gppo');trace=json.loads((data/'physical_joint_trace.json').read_text());m=joint_completion_metric(trace,300.);assert len(trace)==3000 and m==x['summary']['secondary_joint_metric'] and x['source_stable'] and x['optimizer_updates']==0
  assert x['checkpoint_sha256']==P['models'].get(a,P['models']['round8'])['sha256']
  initial.append(json.loads((data/'initial.json').read_text()));hidden.append(json.loads((data/'private_target_identity_diagnostic_only.json').read_text()));cfg=json.loads((data/'config.json').read_text());cfg['task_scheduler'].pop('extended_checkpoint',None);cfg['task_scheduler'].pop('extended_checkpoint_sha256',None);configs.append(cfg)
  arms[a]=dict(seconds=m['secondary_joint_completion_seconds'],success=m['joint_success'],safe=m['whole_window_safe'],physical_ticks=len(trace))
 assert all(z==initial[0] for z in initial) and all(z==hidden[0] for z in hidden) and all(z==configs[0] for z in configs)
 rows.append(dict(case=c,arms=arms,complete=len(arms)==len(P['arms'])))
complete=all(r['complete'] for r in rows);stats={}
if complete:
 for name,subset in [('all',rows),('30',[r for r in rows if r['case']['uavs']==30]),('60',[r for r in rows if r['case']['uavs']==60])]:
  totals={a:sum(r['arms'][a]['seconds'] for r in subset) for a in P['arms']};stats[name]=dict(totals=totals,donor_time_reduction_vs_unfiltered_percent=100*(1-totals['donor']/totals['unfiltered']),donor_time_reduction_vs_rule_percent=100*(1-totals['donor']/totals['rule']),all_success_safe=all(v['success'] and v['safe'] for r in subset for v in r['arms'].values()))
r=dict(status='COMPLETE' if complete else 'RUNNING',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Four previously unseen seeds, one80mmap,30/60 static/dynamic;cross-scenario donor validation,not original8seed acceptance',rows=rows,stats=stats,head=head,source_files_unchanged=len(S),original_checkpoints_unchanged=True,production_edits=False,training=False,commit=False,push=False);(O/'verified_results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
