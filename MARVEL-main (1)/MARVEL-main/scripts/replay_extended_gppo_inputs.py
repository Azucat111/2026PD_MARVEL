"""Replay saved actual actor inputs without running simulation or training."""
from pathlib import Path
import argparse,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from integrations.gppo.gppo_model import GPPOActorCritic
from extended_gppo_heading import HEADING_PROTOCOL
from train_extended_gppo import PROTOCOL

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--out',required=True);p.add_argument('--temperature',type=float,default=1.)
    args=p.parse_args();torch.set_num_threads(1)
    cpfile=Path(args.checkpoint);file=Path(args.inputs)
    cp=torch.load(cpfile,map_location='cpu',weights_only=False)
    if cp.get('protocol_version') not in (PROTOCOL,HEADING_PROTOCOL):raise ValueError('Unsupported experimental protocol')
    model=GPPOActorCritic(edge_feature_dim=6 if cp['protocol_version']==HEADING_PROTOCOL else 3)
    model.load_state_dict(cp['model'],strict=True);model.eval();rows=[]
    for record in json.loads(file.read_text()):
        batch={k:torch.tensor(v,dtype=torch.bool if 'mask' in k or k in ('capability_matrix','precedence_matrix') else torch.float32)
               for k,v in record['batch'].items()}
        with torch.no_grad():output=model(**batch)
        action=record['action'];nt,nu=batch['action_mask'].shape[1:]
        t,u=divmod(action,nu)
        if bool(batch['action_mask'][0,t,u]):raise ValueError('Saved invalid action')
        ids_match=record['chosen']=={'uav_id':record['uav_ids'][u],'subtask_id':record['task_ids'][t]}
        value_error=abs(float(output['value'][0])-record['value']) if record['value'] is not None else 0.
        if record['log_prob'] is None:
            choice_match=action==int(output['flat_logits'][0].argmax());prob_error=None
        else:
            dist=torch.distributions.Categorical(logits=output['flat_logits'][0]/args.temperature)
            prob_error=abs(float(dist.log_prob(torch.tensor(action)))-record['log_prob'])
            choice_match=prob_error<1e-4
        rows.append(dict(time=record['time'],ids_match=ids_match,choice_match=choice_match,
                         value_error=value_error,log_prob_error=prob_error))
    result=dict(checkpoint_sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),graphs=len(rows),rows=rows,
        valid=bool(rows) and all(r['ids_match'] and r['choice_match'] and r['value_error']<1e-4 for r in rows))
    target=ROOT/args.out
    with target.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result),flush=True)
    if not result['valid']:raise RuntimeError('Actor input replay mismatch')

if __name__=='__main__':main()
