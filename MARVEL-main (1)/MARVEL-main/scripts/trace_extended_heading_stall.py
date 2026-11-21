"""Independent real-runtime PPO adaptation, using public TaskGraph actions.
Privileged target events are terminal training reward only, never actor input.
No changes to production inference, physics, shield, frozen model or rule.
"""
from pathlib import Path
import sys,argparse,json,hashlib,time,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import numpy as np
from train_extended_gppo import PROTOCOL as ORIGINAL_PROTOCOL,validate_seed_split
from extended_gppo_heading import HEADING_PROTOCOL as PROTOCOL,HeadingObservations,heading_batch,initialize_heading_model
from integrations.gppo.gppo_model import GPPOActorCritic,batch_single_task_graph
from integrations.gppo.adapter import GPPOAssignment
from integrations.rule_scheduling import choose_rule_assignment

def latency_return(capped_times,decision_time,horizon):
    if horizon<=0 or not capped_times:raise ValueError('Need targets and positive horizon')
    return -sum(max(0.,min(float(t),horizon)-decision_time) for t in capped_times)/len(capped_times)/horizon

def clipped_policy_loss(new_log_prob,old_log_prob,advantage,clip):
    ratio=torch.exp(new_log_prob-old_log_prob)
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()

def main():
    import benchmark_fair_search as bench
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[61011]);p.add_argument('--steps',type=int,default=40)
    p.add_argument('--lr',type=float,default=.00003);p.add_argument('--ppo-epochs',type=int,default=4)
    p.add_argument('--temperature',type=float,default=1.5);args=p.parse_args()
    validate_seed_split(args.seeds,[78901,9148,9149,9150],[101101,202202,303303])
    if args.temperature<=0:raise ValueError('Positive sampling temperature required')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    initial=Path(args.initial).resolve();initial_hash=hashlib.sha256(initial.read_bytes()).hexdigest()
    cp=torch.load(initial,map_location='cpu',weights_only=False)
    if cp.get('protocol_version') not in (PROTOCOL,ORIGINAL_PROTOCOL):raise ValueError('Unsupported experimental checkpoint')
    torch.set_num_threads(1);torch.manual_seed(61000)
    if cp['protocol_version']==ORIGINAL_PROTOCOL:
        model=initialize_heading_model(cp['model'],'cuda')
    else:
        model=GPPOActorCritic(edge_feature_dim=6).cuda();model.load_state_dict(cp['model'],strict=True);model.eval()
    # Fresh reward scale requires a fresh value output; retain graph encoder/actor warm start.
    if cp.get('training_stage')!='online PPO':
        torch.nn.init.zeros_(model.critic.value[-1].weight);torch.nn.init.zeros_(model.critic.value[-1].bias)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    if cp.get('protocol_version')==PROTOCOL and cp.get('training_stage')=='online PPO' and 'optimizer' in cp:
        optimizer.load_state_dict(cp['optimizer'])
        for group in optimizer.param_groups:group['lr']=args.lr
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
             for parent in ('utils','integrations') for f in (ROOT/parent).rglob('*.py')}
    audit_paths=['scripts/trace_extended_heading_stall.py','scripts/extended_gppo_heading.py','scripts/train_extended_gppo.py','scripts/benchmark_fair_search.py',
        'scripts/filter_scheduler_candidates.py','configs/scenarios/extended_live_gppo.yaml',
        'configs/scenarios/assets/extended_live_gppo.png','load_model/MARVEL/checkpoint.pth']
    sources.update({rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in audit_paths})
    frozen_hash=hashlib.sha256(bench.CHECKPOINT.read_bytes()).hexdigest()
    manifest=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frozen_checkpoint_sha256=frozen_hash,args=vars(args),protocol=PROTOCOL,training_stage='online PPO true target latency',
        initial_sha256=initial_hash,source_hashes=sources,device=torch.cuda.get_device_name(),
        final_reserved_seeds=[101101,202202,303303],
        reward='negative remaining true capped detection latency / target_count / horizon; collision penalty',
        actor_input='public normalized graph plus delayed self-sensor heading relative to public task prior; hidden events only reward')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    original_adapter=bench.MARVELPolicyAdapter;history=[]
    base_runtime=bench.SimulationRuntime;trace=[]
    class TraceRuntime(base_runtime):
        def step(self,actions):
            observations,info=super().step(actions)
            scheduler=getattr(self,'high_level_scheduler',None)
            if scheduler is not None:
                heats={int(h.heat_id):np.asarray(h.position).tolist() for h in scheduler.heat_points}
                selected=set(scheduler.search_assignments.values())
                trace.append(dict(step=self.current_step,search_assignments=dict(scheduler.search_assignments),
                    serviced=sorted(scheduler.serviced_heat_ids),public_heats=heats,
                    robots=[dict(uid=int(robot.robot_id),position=robot.position.tolist(),heading=float(robot.heading),
                        action_position=np.asarray(actions[i][0]).tolist(),action_heading=float(actions[i][1]),
                        observation=observations[int(robot.robot_id)])
                        for i,robot in enumerate(self.robots) if robot.robot_id in selected]))
            return observations,info
    bench.SimulationRuntime=TraceRuntime
    for episode,seed in enumerate(args.seeds):
        transitions=[]
        class TrainingAdapter(original_adapter):
            def get_actions(self,observations):
                self.heading_observations.observe(observations)
                return super().get_actions(observations)
            def setup(self):
                result=super().setup();inference=self.scheduler.allocator.adapter
                self.heading_observations=HeadingObservations(self.scheduler.profile.stale_age_fraction)
                def sample(graph,deterministic=True):
                    if graph.num_uavs==0 or graph.num_subtasks==0:return None
                    batch=heading_batch(graph,self.heading_observations,'cuda')
                    with torch.no_grad():
                        output=model(**batch)
                        if not bool(output['has_valid_action'][0]):return None
                        dist=torch.distributions.Categorical(logits=output['flat_logits'][0]/args.temperature)
                        action=dist.sample();flat=int(action);t,u=divmod(flat,graph.num_uavs)
                        teacher=choose_rule_assignment(graph)
                        transitions.append(dict(batch=batch,action=flat,old_log_prob=float(dist.log_prob(action)),
                            value=float(output['value'][0]),time=float(graph.current_time),teacher=teacher.flat_action))
                    return GPPOAssignment(flat,t,u,int(graph.subtasks[t].subtask_id),int(graph.uav_states[u].uav_id),float(output['value'][0]))
                inference.assign=sample
                return result
        bench.MARVELPolicyAdapter=TrainingAdapter
        folder=out/f'episode_{episode:03d}';folder.mkdir()
        try:summary=bench.run(folder,'gppo',30 if episode%2==0 else 60,seed,args.steps,physical_coverage=True,stop_on_completion=True)
        finally:bench.MARVELPolicyAdapter=original_adapter
        (out/'trace.json').write_text(json.dumps(trace,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
        if not summary['valid_for_comparison']:raise RuntimeError('Incomplete/failed runtime episode; no update')
        if not transitions:raise RuntimeError('No allocator transitions')
        times=summary['metric']['capped_detection_seconds'];horizon=summary['metric']['horizon_seconds']
        collision_penalty=.1*sum(summary['collision_types'].values())
        returns=[latency_return(times,tr['time'],horizon)-collision_penalty for tr in transitions]
        old_values=torch.tensor([tr['value'] for tr in transitions],device='cuda')
        targets=torch.tensor(returns,device='cuda');advantages=(targets-old_values).detach()
        losses=[]
        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True);loss=0.
            for i,tr in enumerate(transitions):
                output=model(**tr['batch']);dist=torch.distributions.Categorical(logits=output['flat_logits'][0]/args.temperature)
                newlog=dist.log_prob(torch.tensor(tr['action'],device='cuda'))
                policy=clipped_policy_loss(newlog,torch.tensor(tr['old_log_prob'],device='cuda'),advantages[i],.2)
                value=.25*(output['value'][0]-targets[i]).square()
                anchor=.05*torch.nn.functional.cross_entropy(output['flat_logits'],torch.tensor([tr['teacher']],device='cuda'))
                loss=loss+policy+value+anchor-.002*dist.entropy()
            loss=loss/len(transitions)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        row=dict(episode=episode,seed=seed,uavs=summary['uavs'],transitions=len(transitions),metric=summary['metric'],
                 collision_types=summary['collision_types'],returns=returns,losses=losses)
        history.append(row);print('PPO_UPDATE '+json.dumps(row),flush=True)
        torch.save(dict(protocol_version=PROTOCOL,training_stage='online PPO',model=model.state_dict(),
            optimizer=optimizer.state_dict(),episode=episode,initial_sha256=initial_hash),out/f'checkpoint_{episode:03d}.pth')
        (out/'progress.json').write_text(json.dumps(history,indent=2))
        assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest for rel,digest in sources.items())
    assert hashlib.sha256(initial.read_bytes()).hexdigest()==initial_hash
    assert hashlib.sha256(bench.CHECKPOINT.read_bytes()).hexdigest()==frozen_hash
    (out/'complete.json').write_text(json.dumps(dict(episodes=len(history),initial_unchanged=True,acceptance_claim=False),indent=2))

if __name__=='__main__':main()
