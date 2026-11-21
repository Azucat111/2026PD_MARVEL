"""Bounded failed-episode reconstruction with exact recorded graph/action checks."""
from pathlib import Path
import argparse,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from integrations.gppo.gppo_model import GPPOActorCritic
from integrations.gppo.adapter import GPPOAssignment
from extended_gppo_heading import HEADING_PROTOCOL,HeadingObservations,heading_batch
from gppo_motion_capture import capture_motion_frame,append_motion_frame

def main():
    import benchmark_fair_search as bench
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--repair-regression',action='store_true');args=p.parse_args()
    training=ROOT/args.training;manifest=json.loads((training/'manifest.json').read_text())
    episode=training/f'episode_{args.episode:03d}';expected=json.loads((episode/'actor_inputs.json').read_text())
    history=json.loads((training/'progress.json').read_text());row=history[args.episode]
    checkpoint=training/f'checkpoint_{args.episode-1:03d}.pth'
    if args.episode<=0:raise ValueError('Use explicit prior checkpoint; episode zero unsupported')
    current_sources={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in manifest['source_hashes']}
    changed=[rel for rel,digest in manifest['source_hashes'].items() if current_sources[rel]!=digest]
    if changed and (not args.repair_regression or set(changed)!={'integrations/gppo/scheduler.py'}):raise ValueError('Unexpected changed sources: '+str(changed))
    for rel in ('scripts/reconstruct_failed_heading_episode.py','scripts/gppo_motion_capture.py'):
        current_sources[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    for rel in current_sources:
        if Path(rel).suffix in ('.py','.yaml','.png'):
            saved=out/'source_snapshot'/rel;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes((ROOT/rel).read_bytes())
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if cp['protocol_version']!=HEADING_PROTOCOL:raise ValueError('Wrong protocol')
    torch.set_num_threads(1);torch.manual_seed(61000)
    model=GPPOActorCritic(edge_feature_dim=6).cuda();model.load_state_dict(cp['model']);model.eval()
    base=bench.MARVELPolicyAdapter;checked=[];divergences=[]
    class ReplayAdapter(base):
        def get_actions(self,observations):
            self.headings.observe(observations)
            actions=super().get_actions(observations)
            append_motion_frame(out/'gppo/mission_frames.jsonl',capture_motion_frame(self.runtime,self.scheduler,observations,actions))
            return actions
        def setup(self):
            result=super().setup();self.headings=HeadingObservations(self.scheduler.profile.stale_age_fraction)
            def assign(graph,deterministic=True):
                if not graph.num_uavs or not graph.num_subtasks:return None
                batch=heading_batch(graph,self.headings,'cuda')
                with torch.no_grad():
                    output=model(**batch)
                    if not bool(output['has_valid_action'][0]):return None
                    dist=torch.distributions.Categorical(logits=output['flat_logits'][0]/manifest['args']['temperature'])
                    action=dist.sample();flat=int(action);t,u=divmod(flat,graph.num_uavs)
                    index=len(checked)
                    if index>=len(expected) and not args.repair_regression:raise ValueError('Unexpected additional allocator action')
                    saved=expected[index] if index<len(expected) else None
                    ids=[x.uav_id for x in graph.uav_states];tasks=[x.subtask_id for x in graph.subtasks]
                    match=saved is not None and ids==saved['uav_ids'] and tasks==saved['task_ids'] and abs(graph.current_time-saved['time'])<=1e-8
                    error=0.
                    if match:
                        for name,tensor in batch.items():
                            previous=torch.tensor(saved['batch'][name],device=tensor.device,dtype=tensor.dtype)
                            if tensor.shape!=previous.shape:match=False;break
                            if tensor.dtype==torch.bool:
                                if not torch.equal(tensor,previous):match=False;break
                            elif tensor.numel():error=max(error,float((tensor-previous).abs().max()))
                        match=match and error<=1e-6 and flat==saved['action']
                    if not match:
                        divergences.append(dict(index=index,time=graph.current_time,action=flat,max_tensor_error=error))
                        if not args.repair_regression:raise ValueError('Recorded graph/action mismatch')
                    checked.append(dict(index=index,time=graph.current_time,action=flat,max_tensor_error=error,original_match=match))
                return GPPOAssignment(flat,t,u,int(tasks[t]),int(ids[u]),float(output['value'][0]))
            self.scheduler.allocator.adapter.assign=assign
            return result
    bench.MARVELPolicyAdapter=ReplayAdapter
    (out/'manifest.json').write_text(json.dumps(dict(source_training=str(training),episode=args.episode,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_hashes=current_sources,original_training_source_hashes=manifest['source_hashes'],
        purpose='repair regression on prior failure, not acceptance' if args.repair_regression else 'diagnostic reconstruction, not another acceptance sample',source_changes=changed,sampled_actions_verified=not args.repair_regression),indent=2))
    try:summary=bench.run(out,'gppo',row['uavs'],row['seed'],int(manifest['args']['steps']),physical_coverage=True,stop_on_completion=True)
    finally:bench.MARVELPolicyAdapter=base
    original=json.loads((episode/'gppo/summary.json').read_text())
    same_metrics=summary['metric']==original['metric']
    same_collisions=summary['collision_types']==original['collision_types']
    for rel,digest in current_sources.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest
    complete=not args.repair_regression and not divergences and len(checked)==len(expected) and same_metrics and same_collisions and not summary['policy_execution_failures']
    (out/'reconstruction.json').write_text(json.dumps(dict(exact_recorded_inputs_actions_and_outcomes=complete,
        checked=checked,same_metrics=same_metrics,same_collisions=same_collisions,acceptance_claim=False),indent=2))
    if args.repair_regression:
        repaired=summary['metric']['detected_targets']==summary['metric']['total_targets'] and not summary['collision_types'] and not summary['policy_execution_failures']
        (out/'repair_regression.json').write_text(json.dumps(dict(repaired=repaired,original_matched_actions=sum(c['original_match'] for c in checked),divergences=divergences,metric=summary['metric'],acceptance_claim=False),indent=2))
    elif not complete:raise RuntimeError('Reconstruction mismatch; do not infer original hidden states')
if __name__=='__main__':main()
