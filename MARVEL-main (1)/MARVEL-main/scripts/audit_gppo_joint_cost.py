"""Observed-cost diagnostic only; never installs a production planner."""
from pathlib import Path
import json,argparse

def two_task_costs(costs,masks):
 if len(costs)!=2 or len(masks)!=2:raise ValueError('Exactly two task rows required')
 if len(costs[0])!=len(costs[1]) or len(costs[0])!=len(masks[0]) or len(costs[1])!=len(masks[1]):raise ValueError('Incompatible shape')
 pairs=[(float(costs[0][u])+float(costs[1][v]),u,v) for u in range(len(costs[0])) for v in range(len(costs[1])) if u!=v and not masks[0][u] and not masks[1][v]]
 if not pairs:return None
 first=min((float(costs[t][u]),t,u) for t in range(2) for u in range(len(costs[t])) if not masks[t][u])
 t,u=first[1:];second=[(float(costs[1-t][v]),v) for v in range(len(costs[1-t])) if v!=u and not masks[1-t][v]]
 if not second:return dict(optimal=min(pairs)[0],greedy=None)
 return dict(optimal=min(pairs)[0],greedy=first[0]+min(second)[0])

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];rows=[]
 for f in sorted((root/a.run).glob('episode_*/gppo/decisions.json')):
  for g in json.loads(f.read_text()):
   priority=max((g['raw_task_features'][t][0] for t in range(len(g['task_ids'])) if any(not m for m in g['mask'][t])),default=None)
   tasks=[t for t in range(len(g['task_ids'])) if g['raw_task_features'][t][0]==priority and any(not m for m in g['mask'][t])]
   if len(tasks)!=2:continue
   result=two_task_costs([g['observed_total_time'][t] for t in tasks],[g['mask'][t] for t in tasks])
   if result:rows.append(dict(episode=f.parent.parent.name,time=g['current_time'],**result))
 comparable=[r for r in rows if r['greedy'] is not None]
 result=dict(graphs=len(rows),greedy_dead_ends=sum(r['greedy'] is None for r in rows),improvable=sum(r['greedy']>r['optimal']+1e-6 for r in comparable),total_greedy=sum(r['greedy'] for r in comparable),total_optimal=sum(r['optimal'] for r in comparable),rows=rows,caveat='Only two-row instantaneous observed-cost bound. Not hidden-map truth, actual trajectories or project efficiency.')
 with (root/a.out).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
