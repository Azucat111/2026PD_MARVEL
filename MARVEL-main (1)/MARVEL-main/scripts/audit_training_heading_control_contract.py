from pathlib import Path
import sys,json,argparse,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.heading_features import HeadingObservations

def audit_record(record):
 state=record['public_heading_control_snapshot']
 interval=float(state['interval_steps'])*float(state['physics_dt'])
 assert interval>0 and math.isfinite(interval)
 expected_index=int(math.floor(float(record['time'])/interval+1e-9))
 assert state['control_index']==expected_index, (state['control_index'],expected_index)
 observer=HeadingObservations(state['age_fraction'])
 observer.current={int(k):float(v) for k,v in state['current'].items()}
 observer.previous={int(k):float(v) for k,v in state['previous'].items()}
 actual=np.asarray(record['batch']['edge_features'],dtype=np.float32)[0,:,:,3:]
 expected=np.zeros_like(actual)
 for t,task in enumerate(state['public_task_positions']):
  if task is None:continue
  for u,(uid,position) in enumerate(zip(record['uav_ids'],state['public_uav_positions'])):
   angle=observer.heading(int(uid))
   if angle is None or position is None:continue
   bearing=math.degrees(math.atan2(task[1]-position[1],task[0]-position[0]))%360.
   relative=math.radians((bearing-angle+180.)%360.-180.)
   expected[t,u]=[math.cos(relative),math.sin(relative),1.]
 error=float(np.max(np.abs(actual-expected))) if actual.size else 0.
 assert error<1e-6,error
 return error

def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--episodes',type=int,required=True);p.add_argument('--out',required=True);args=p.parse_args()
 errors=[];records=0
 for i in range(args.episodes):
  items=json.loads((ROOT/args.folder/f'episode_{i:03d}/actor_inputs.json').read_text())
  for record in items:errors.append(audit_record(record));records+=1
 assert records>0
 result=dict(episodes=args.episodes,records=records,production_control_clock_verified=True,public_graph_and_heading_only=True,max_heading_feature_error=max(errors),passed=True,acceptance_claim=False)
 with (ROOT/args.out).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
