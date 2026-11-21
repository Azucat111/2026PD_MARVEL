"""Screen a candidate on saved public actor inputs; no new trajectory or gain claim."""
from pathlib import Path
import argparse,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from integrations.gppo.gppo_model import GPPOActorCritic
from extended_gppo_heading import HEADING_PROTOCOL

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(1)
    cpfile=Path(args.checkpoint);cp=torch.load(cpfile,map_location='cpu',weights_only=False)
    if cp.get('protocol_version')!=HEADING_PROTOCOL:raise ValueError('Expected heading candidate')
    model=GPPOActorCritic(edge_feature_dim=6);model.load_state_dict(cp['model'],strict=True);model.eval()
    rows=[];sources=[]
    for directory in args.cases:
        folder=ROOT/directory;inputs=folder/'gppo/actor_inputs.json';decisions=folder/'gppo/decisions.json'
        records=json.loads(inputs.read_text());graphs=[g for g in json.loads(decisions.read_text()) if g['chosen'] is not None]
        if len(records)!=len(graphs):raise ValueError('Actual actor/graph record count differs')
        sources.append(dict(case=directory,input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
                            graph_sha256=hashlib.sha256(decisions.read_bytes()).hexdigest()))
        for record,graph in zip(records,graphs):
            if record['uav_ids']!=graph['uav_ids'] or record['task_ids']!=graph['task_ids'] or record['chosen']!=graph['chosen']:
                raise ValueError('Actual input graph/action alignment mismatch')
            batch={k:torch.tensor(v,dtype=torch.bool if 'mask' in k or k in ('capability_matrix','precedence_matrix') else torch.float32)
                   for k,v in record['batch'].items()}
            with torch.no_grad():output=model(**batch)
            action=int(output['flat_logits'][0].argmax());nt,nu=batch['action_mask'].shape[1:];t,u=divmod(action,nu)
            if bool(batch['action_mask'][0,t,u]):raise ValueError('Candidate chose masked action')
            oldt,oldu=divmod(record['action'],nu)
            rows.append(dict(case=directory,time=record['time'],changed=action!=record['action'],
                chosen=dict(uav_id=record['uav_ids'][u],subtask_id=record['task_ids'][t]),
                old_chosen=record['chosen'],observed_cost=graph['observed_total_time'][t][u],
                old_observed_cost=graph['observed_total_time'][oldt][oldu],
                priority_violation=graph['raw_task_features'][t][0]<max(graph['raw_task_features'][i][0]
                    for i in range(nt) if any(not x for x in graph['mask'][i]))))
    result=dict(checkpoint_sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest(),sources=sources,graphs=len(rows),
        changed=sum(r['changed'] for r in rows),priority_violations=sum(r['priority_violation'] for r in rows),rows=rows,
        caveat='Shadow decisions on existing observations only. Not a new trajectory and not measured efficiency gain.')
    with (ROOT/args.out).open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
