"""Evidence audit for corrected-unit dynamic recovery, not efficiency acceptance."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];PAIR=ROOT/'artifacts/extended_corrected_units_joint_30_dynamic_9162_20261010_attempt3'
m=json.loads((PAIR/'manifest.json').read_text())
assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==s for f,s in m['source_hashes'].items())
assert hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest()==m['checkpoint_sha256']
frozen=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
assert hashlib.sha256(frozen.read_bytes()).hexdigest()==m['frozen_checkpoint_sha256']
assert json.loads((PAIR/'gppo/initial.json').read_text())==json.loads((PAIR/'rule/initial.json').read_text())
rows=[]
for arm in ('gppo','rule'):
 s=json.loads((PAIR/arm/'summary.json').read_text());c=json.loads((PAIR/arm/'config.json').read_text());t=json.loads((PAIR/arm/'joint_lifecycle_trace.json').read_text());last=t[-1]
 warning=next(o for o in c['dynamic_obstacles'] if o['id']=='natural_expanding_warning')
 # Actual resolved physics parameters, not advertised mission-rate units.
 rate=warning['params']['expansion_rate']/(c['_gppo_profile']['physics_step_ms']/1000)
 assert abs(rate-.2)<1e-12
 assert s['mission_steps']==120 and s['metric']['detected_targets']==4 and not s['collision_types']
 assert last['task_summary']['T3_relay']['progress']>=.9 and not last['safety_active'] and last['no_fly_violations']==0
 assert last['escapes']==last['safety_records'] and last['safety_records']>0
 rows.append(dict(arm=arm,physical_warning_mps=rate,full_horizon=120,search_found=4,mean_search_seconds=s['metric']['mean_capped_seconds'],collisions=0,final_radio_ratio=last['task_summary']['T3_relay']['progress'],first_threshold_step=next(r['step'] for r in t if r['task_summary']['T3_relay']['progress']>=.9),safety_triggers=last['safety_records'],escapes=last['escapes'],no_fly_violations=0,final_helpers=last['relay_assignments'],relay_release_count=last['relay_release_count']))
result=dict(rows=rows,source_candidate_frozen_hashes_preserved=True,same_initial_state=True,search_efficiency_gain=rows[1]['mean_search_seconds']/rows[0]['mean_search_seconds']-1,passed=True,acceptance_claim=False,scope='30-UAV corrected .2m/s dynamic development H120; native source regressions and controlled 60 Relay are separate evidence. Reserved final seeds unused.')
with (ROOT/'artifacts/extended_dynamic_relay_warning_repair_20261010_attempt1/full_pair_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
