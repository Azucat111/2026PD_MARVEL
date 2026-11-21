"""Read-only policy exploration audit on actual public rollout tensors."""
from pathlib import Path
import argparse,json,sys,math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from integrations.gppo.gppo_model import GPPOActorCritic

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--temperature',type=float,default=2.5);args=p.parse_args()
 torch.set_num_threads(1)
 cp=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=False)
 model=GPPOActorCritic(edge_feature_dim=6);model.load_state_dict(cp['model'],strict=True);model.eval()
 rows=[]
 for folder in sorted((ROOT/args.run).glob('episode_*')):
  inp=folder/'actor_inputs.json';dec=folder/'gppo/decisions.json'
  if not inp.exists() or not dec.exists():continue
  records=json.loads(inp.read_text());graphs=[g for g in json.loads(dec.read_text()) if g['chosen'] is not None]
  if len(records)!=len(graphs):raise ValueError('Unaligned records')
  for r,g in zip(records,graphs):
   batch={k:torch.tensor(v,dtype=torch.bool if 'mask' in k or k in ('capability_matrix','precedence_matrix') else torch.float32) for k,v in r['batch'].items()}
   with torch.no_grad():o=model(**batch)
   dist=torch.distributions.Categorical(logits=o['flat_logits'][0]/args.temperature)
   n=int((~batch['action_mask']).sum())
   chosen=r['action']; nt,nu=batch['action_mask'].shape[1:]
   legal=[(-g['raw_task_features'][t][0],g['observed_total_time'][t][u],g['task_ids'][t],g['uav_ids'][u],t*nu+u) for t in range(nt) for u in range(nu) if not g['mask'][t][u]]
   teacher=min(legal)[-1]
   rows.append(dict(episode=folder.name,time=r['time'],valid=n,entropy=float(dist.entropy()),max_entropy=math.log(n),max_probability=float(dist.probs.max()),teacher_probability=float(dist.probs[teacher]),sample_differs_teacher=chosen!=teacher,recorded_log_prob=r['log_prob']))
 result=dict(graphs=len(rows),sample_differs_teacher=sum(x['sample_differs_teacher'] for x in rows),mean_max_probability=sum(x['max_probability'] for x in rows)/len(rows),mean_teacher_probability=sum(x['teacher_probability'] for x in rows)/len(rows),mean_normalized_entropy=sum(x['entropy']/x['max_entropy'] if x['max_entropy'] else 0 for x in rows)/len(rows),rows=rows,caveat='Latest model on older public inputs; recorded actions sampled with then-current model. No causal gain claim.')
 with (ROOT/args.out).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
