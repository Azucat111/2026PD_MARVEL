from pathlib import Path
import json,sys,hashlib,subprocess,datetime
R=Path.cwd();O=Path(__file__).resolve().parent;OLD=R/'artifacts/gppo_three_hour_20261110';sys.path.insert(0,str(R/'scripts'))
from gppo_joint_completion_metric import joint_completion_metric
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
S=json.loads((OLD/'corrected_source_hashes.json').read_text());assert all(sha(R/f)==h for f,h in S.items())
P0=json.loads((OLD/'plan.json').read_text());assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==P0['head'];assert all(sha(R/m['path'])==m['sha256'] for m in P0['models'].values())
assets=json.loads((OLD/'execution_assets.json').read_text());assert all(sha(Path(f))==v['sha256'] for f,v in assets['files'].items())
result=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Fixed-training-case learnability and prespecified new-seed diagnostic, one connected80m map; no overall15percent acceptance or500m claim',source_files_unchanged=len(S),original_checkpoints_unchanged=True,execution_assets_unchanged=len(assets['files']),head=P0['head'],production_edits_this_turn=False,commit=False,push=False,stages={})
for name in ['evaluation','transfer']:
 D=O/name
 if not (D/'plan.json').exists():result['stages'][name]=dict(status='NOT_STARTED');continue
 P=json.loads((D/'plan.json').read_text());assert sha(D/'plan.json')==(D/'protocol_sha256.txt').read_text().strip()
 assert all(sha(R/f)==h for f,h in P['source_hashes'].items());assert all(sha(R/m['path'])==m['sha256'] for m in P['models'].values())
 proof_path=R/P['update_proof_path'];U=json.loads(proof_path.read_text());assert sha(proof_path)==P['update_proof_sha256'];assert U['candidate_sha256']==P['models']['candidate']['sha256']==sha(R/U['candidate'])
 rows=[]
 for i,c in enumerate(P['cases']):
  arms={};initials=[];layouts=[];configs=[]
  for a in P['arms']:
   folder=D/'data'/f"case_{i:02d}_{c['seed']}_{c['uavs']}_{'dynamic' if c['dynamic'] else 'static'}"/(a+'_original');f=folder/'complete.json'
   if not f.exists():continue
   x=json.loads(f.read_text());data=folder/('rule' if a=='rule' else 'gppo');trace=json.loads((data/'physical_joint_trace.json').read_text());m=joint_completion_metric(trace,300.);assert len(trace)==3000 and m==x['summary']['secondary_joint_metric'] and x['source_stable'];assert x['checkpoint_sha256']==P['models'].get(a,P['models']['round8'])['sha256'];assert x['optimizer_updates']==0
   initials.append(json.loads((data/'initial.json').read_text()));layouts.append(json.loads((data/'private_target_identity_diagnostic_only.json').read_text()));cfg=json.loads((data/'config.json').read_text());cfg['task_scheduler'].pop('extended_checkpoint',None);cfg['task_scheduler'].pop('extended_checkpoint_sha256',None);configs.append(cfg)
   C=m['secondary_joint_completion_seconds'];post=[r for r in trace if r['time']>=C];arms[a]=dict(seconds=C,success=m['joint_success'],safe=m['whole_window_safe'],full_physical_ticks=len(trace),post_completion_disconnect_seconds=(sum(r['connectivity']<.9 for r in post)*.1 if m['joint_success'] else None),final_connectivity=trace[-1]['connectivity'])
  assert all(z==initials[0] for z in initials) and all(z==layouts[0] for z in layouts) and all(z==configs[0] for z in configs)
  rows.append(dict(case=c,arms=arms,complete=len(arms)==3))
 complete=all(r['complete'] for r in rows);totals={};stats={}
 if complete:
  totals={a:sum(r['arms'][a]['seconds'] for r in rows) for a in P['arms']};stats=dict(totals=totals,time_reduction_vs_parent_percent=100*(1-totals['candidate']/totals['parent']),time_reduction_vs_rule_percent=100*(1-totals['candidate']/totals['rule']),all_success_safe=all(m['success'] and m['safe'] for r in rows for m in r['arms'].values()))
 result['stages'][name]=dict(status='COMPLETE' if complete else 'PARTIAL',rows=rows,stats=stats)
result['diagnostic_complete']=all(result['stages'][n]['status']=='COMPLETE' for n in ['evaluation','transfer']);result['overall_project_15_percent_acceptance']=None
(O/'verified_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))



