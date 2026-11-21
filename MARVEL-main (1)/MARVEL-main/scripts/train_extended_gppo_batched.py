"""Independent real-runtime PPO adaptation, using public TaskGraph actions.
Privileged target events are terminal training reward only, never actor input.
No changes to production inference, physics, shield, frozen model or rule.
"""
from pathlib import Path
import sys,argparse,json,hashlib,time,subprocess,math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from train_extended_gppo import PROTOCOL as ORIGINAL_PROTOCOL,validate_seed_split
from integrations.gppo.heading_features import HEADING_PROTOCOL as PROTOCOL,HeadingObservations,heading_batch
from extended_gppo_heading import initialize_heading_model
from audit_training_heading_control_contract import audit_record as audit_heading_control_record
from integrations.gppo.gppo_model import GPPOActorCritic,batch_single_task_graph
from integrations.gppo.adapter import GPPOAssignment
from integrations.rule_scheduling import choose_rule_assignment
from gppo_training_records import actor_input_record
from gppo_motion_capture import capture_motion_frame,append_motion_frame
from gppo_rollout_batch import FrozenBehaviorBatch,validate_reward_horizon
from gppo_benchmark_domains import load_domain,scenario_paths
from gppo_joint_reward import remaining_connectivity_deficit,validate_reward_protocol

def latency_return(capped_times,decision_time,horizon):
    if horizon<=0 or not capped_times:raise ValueError('Need targets and positive horizon')
    return -sum(max(0.,min(float(t),horizon)-decision_time) for t in capped_times)/len(capped_times)/horizon

def clipped_policy_loss(new_log_prob,old_log_prob,advantage,clip):
    ratio=torch.exp(new_log_prob-old_log_prob)
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()

def reset_value_head(model):
    torch.nn.init.zeros_(model.critic.value[-1].weight)
    torch.nn.init.zeros_(model.critic.value[-1].bias)

def bind_production_heading_observer(scheduler):
    """Let the production scheduler own control-clock observation history."""
    observer = HeadingObservations(scheduler.profile.stale_age_fraction)
    scheduler.allocator.adapter.observe = observer.observe
    # A newly bound observer needs its own initial control snapshot.
    if hasattr(scheduler, '_last_heading_observation_control_index'):
        del scheduler._last_heading_observation_control_index
    return observer

def main():
    import benchmark_fair_search as bench
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[61011,61012,61013,61014]);p.add_argument('--steps',type=int,default=80)
    p.add_argument('--lr',type=float,default=.00003);p.add_argument('--ppo-epochs',type=int,default=4)
    p.add_argument('--temperature',type=float,default=1.5)
    p.add_argument('--reset-critic',action='store_true',help='Fresh value head and optimizer when changing reward scale')
    p.add_argument('--rollout-batch-size',type=int,default=8);p.add_argument('--target-kl',type=float,default=.03)
    p.add_argument('--anchor-weight',type=float,default=.005);p.add_argument('--entropy-weight',type=float,default=.01)
    p.add_argument('--scenario');p.add_argument('--balanced-domains',action='store_true');p.add_argument('--warning-rate-mps',type=float,default=.2)
    p.add_argument('--joint-rollout',action='store_true');p.add_argument('--relay-reward-weight',type=float,default=.25)
    args=p.parse_args()
    if not math.isfinite(args.relay_reward_weight) or args.relay_reward_weight<0:raise ValueError('Finite nonnegative Relay reward weight required')
    if args.balanced_domains and (not math.isfinite(args.warning_rate_mps) or args.warning_rate_mps<=0):raise ValueError('Balanced dynamic domain needs positive finite physical warning speed')
    if not math.isfinite(args.target_kl) or args.target_kl<=0:raise ValueError('Positive finite target KL required')
    collector=FrozenBehaviorBatch(args.rollout_batch_size)
    if args.anchor_weight<0 or args.entropy_weight<0:raise ValueError('Nonnegative regularization weights required')
    validate_seed_split(args.seeds,[78901,9148,9149,9150,9151,9152,9153,9154,9155,9156,9157,9158,9159,9160,9161,9162,9163,9164,9165],[101101,202202,303303])
    if args.temperature<=0:raise ValueError('Positive sampling temperature required')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    initial=Path(args.initial).resolve();initial_hash=hashlib.sha256(initial.read_bytes()).hexdigest()
    cp=torch.load(initial,map_location='cpu',weights_only=False)
    validate_reward_horizon(cp,args.steps,args.reset_critic)
    validate_reward_protocol(cp,args.joint_rollout,args.reset_critic,args.relay_reward_weight)
    reward_protocol="joint_physical_v1" if args.joint_rollout else "search_latency_v1"
    if cp.get('protocol_version') not in (PROTOCOL,ORIGINAL_PROTOCOL):raise ValueError('Unsupported experimental checkpoint')
    torch.set_num_threads(1);torch.manual_seed(61000)
    if cp['protocol_version']==ORIGINAL_PROTOCOL:
        model=initialize_heading_model(cp['model'],'cuda')
    else:
        model=GPPOActorCritic(edge_feature_dim=6).cuda();model.load_state_dict(cp['model'],strict=True);model.eval()
    # Fresh reward scale requires a fresh value output; retain graph encoder/actor warm start.
    if args.reset_critic or cp.get('training_stage')!='online PPO':
        reset_value_head(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    if cp.get('protocol_version')==PROTOCOL and cp.get('training_stage')=='online PPO' and 'optimizer' in cp and not args.reset_critic:
        optimizer.load_state_dict(cp['optimizer'])
        for group in optimizer.param_groups:group['lr']=args.lr
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
             for parent in ('utils','integrations') for f in (ROOT/parent).rglob('*.py')}
    audit_paths=['scripts/train_extended_gppo_heading.py','scripts/train_extended_gppo_batched.py','scripts/gppo_rollout_batch.py','scripts/gppo_motion_capture.py','scripts/extended_gppo_heading.py','scripts/train_extended_gppo.py','scripts/benchmark_fair_search.py',
        'scripts/gppo_training_records.py','scripts/audit_training_heading_control_contract.py','scripts/filter_scheduler_candidates.py','configs/scenarios/extended_live_gppo.yaml',
        'configs/scenarios/assets/extended_live_gppo.png','load_model/MARVEL/checkpoint.pth','parameter.py','test_parameter.py',
        'configs/dynamics/kinematic_simple.yaml','configs/sensors/ideal.yaml','configs/communication/ideal.yaml']
    audit_paths += ['scripts/gppo_benchmark_domains.py','scripts/gppo_joint_reward.py']+[str(f.relative_to(ROOT)) for f in scenario_paths(args.scenario)]
    sources.update({rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in audit_paths})
    frozen_hash=hashlib.sha256(bench.CHECKPOINT.read_bytes()).hexdigest()
    manifest=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frozen_checkpoint_sha256=frozen_hash,args=vars(args),protocol=PROTOCOL,training_stage='batched on-policy PPO true target latency',
        reward_protocol=reward_protocol,
        batch_semantics='same model state for all independent seed rollouts in batch; same team reward',
        domain_cycle=['30 static','60 static','30 dynamic','60 dynamic'] if args.balanced_domains else ['30 static','60 static'],
        warning_rate_mps=args.warning_rate_mps if args.balanced_domains else None,
        initial_sha256=initial_hash,source_hashes=sources,device=torch.cuda.get_device_name(),
        final_reserved_seeds=[101101,202202,303303],
        reward='negative remaining true capped detection latency / target_count / horizon; optional weighted actual radio deficit below .9 over full horizon; collision penalty',
        actor_input='public normalized graph plus delayed self-sensor heading relative to public task prior; hidden events only reward')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for rel in sources:
        if Path(rel).suffix in ('.py','.yaml','.png'):
            saved=out/'source_snapshot'/rel;saved.parent.mkdir(parents=True,exist_ok=True)
            saved.write_bytes((ROOT/rel).read_bytes())
    original_adapter=bench.MARVELPolicyAdapter;original_loader=bench.load_and_validate_scenario;original_runtime=bench.SimulationRuntime;history=[];updates=[]
    for episode,seed in enumerate(args.seeds):
        dynamic=bool(args.balanced_domains and episode%4>=2)
        bench.load_and_validate_scenario=lambda file:load_domain(original_loader,file,args.scenario,dynamic,args.warning_rate_mps)
        transitions=[];episode_runtime=None;episode_scheduler=None;radio_samples=[];collision_times=[]
        class RewardRuntime(original_runtime):
            def step(self,actions):
                obs,info=super().step(actions)
                radio_samples.append((self.current_step*self.dt,float(info["connectivity_ratio"])))
                collision_times.extend([self.current_step*self.dt]*len(info["collisions"]))
                return obs,info
        bench.SimulationRuntime=RewardRuntime
        if collector.episodes==0:
            collector.begin(model)
            torch.save(dict(protocol_version=PROTOCOL,training_stage='online PPO',model=model.state_dict(),
                optimizer=optimizer.state_dict(),training_args=vars(args),reward_protocol=reward_protocol,behavior_sha256=collector.policy_sha256),
                out/f'behavior_batch_{len(updates):03d}.pth')
        class TrainingAdapter(original_adapter):
            def get_actions(self,observations):
                actions=super().get_actions(observations)
                append_motion_frame(folder/'gppo/mission_frames.jsonl',
                    capture_motion_frame(self.runtime,self.scheduler,observations,actions))
                return actions
            def setup(self):
                nonlocal episode_runtime,episode_scheduler
                result=super().setup();episode_runtime=self.runtime;episode_scheduler=self.scheduler;inference=self.scheduler.allocator.adapter
                self.heading_observations=bind_production_heading_observer(self.scheduler)
                def sample(graph,deterministic=True):
                    if graph.num_uavs==0 or graph.num_subtasks==0:return None
                    batch=heading_batch(graph,self.heading_observations,'cuda')
                    with torch.no_grad():
                        output=model(**batch)
                        if not bool(output['has_valid_action'][0]):return None
                        dist=torch.distributions.Categorical(logits=output['flat_logits'][0]/args.temperature)
                        action=dist.sample();flat=int(action);t,u=divmod(flat,graph.num_uavs)
                        teacher=choose_rule_assignment(graph)
                        transitions.append(dict(role=int(graph.subtasks[t].task_type),batch=batch,action=flat,old_log_prob=float(dist.log_prob(action)),
                            value=float(output['value'][0]),time=float(graph.current_time),teacher=teacher.flat_action,
                            record=actor_input_record(batch,time=graph.current_time,action=flat,
                                uav_ids=[u.uav_id for u in graph.uav_states],task_ids=[t.subtask_id for t in graph.subtasks],
                                log_prob=float(dist.log_prob(action)),value=float(output['value'][0]))))
                    transitions[-1]['record']['public_heading_control_snapshot'] = dict(
                        control_index=int(self.scheduler._last_heading_observation_control_index),
                        interval_steps=int(self.scheduler.profile.high_level_interval_steps),
                        physics_dt=float(self.runtime.dt),
                        age_fraction=float(self.heading_observations.age_fraction),
                        current=dict(self.heading_observations.current),
                        previous=dict(self.heading_observations.previous),
                        public_uav_positions=[None if u.position is None else list(u.position) for u in graph.uav_states],
                        public_task_positions=[None if task.position is None else list(task.position) for task in graph.subtasks],
                    )
                    audit_heading_control_record(transitions[-1]['record'])
                    return GPPOAssignment(flat,t,u,int(graph.subtasks[t].subtask_id),int(graph.uav_states[u].uav_id),float(output['value'][0]))
                inference.assign=sample
                return result
        bench.MARVELPolicyAdapter=TrainingAdapter
        folder=out/f'episode_{episode:03d}';folder.mkdir()
        try:summary=bench.run(folder,'gppo',30 if episode%2==0 else 60,seed,args.steps,physical_coverage=True,stop_on_completion=not args.joint_rollout)
        finally:
            bench.MARVELPolicyAdapter=original_adapter
            bench.load_and_validate_scenario=original_loader
            bench.SimulationRuntime=original_runtime
        append_motion_frame(folder/'gppo/mission_frames.jsonl',capture_motion_frame(episode_runtime,episode_scheduler))
        (folder/'actor_inputs.json').write_text(json.dumps([tr['record'] for tr in transitions],indent=2))
        if not summary['valid_for_comparison']:raise RuntimeError('Incomplete/failed runtime episode; no update')
        if args.joint_rollout and summary['mission_steps']!=args.steps:raise RuntimeError('Incomplete joint horizon; no update')
        if not transitions:raise RuntimeError('No allocator transitions')
        (folder/'physical_radio_reward_trace.json').write_text(json.dumps(radio_samples))
        times=summary['metric']['capped_detection_seconds'];horizon=summary['metric']['horizon_seconds']
        collision_penalty=.1*sum(summary['collision_types'].values())
        relay_deficits=[remaining_connectivity_deficit(radio_samples,tr['time'],horizon) if args.joint_rollout else 0. for tr in transitions]
        penalties=[.1*sum(t>tr['time'] for t in collision_times) if args.joint_rollout else collision_penalty for tr in transitions]
        (folder/'physical_collision_reward_times.json').write_text(json.dumps(collision_times))
        returns=[latency_return(times,tr['time'],horizon)-penalty-args.relay_reward_weight*deficit for tr,deficit,penalty in zip(transitions,relay_deficits,penalties)]
        collector.add_episode(model,transitions,returns)
        row=dict(episode=episode,seed=seed,domain='dynamic' if dynamic else 'static',uavs=summary['uavs'],transitions=len(transitions),metric=summary['metric'],
                 collision_types=summary['collision_types'],returns=returns,relay_deficits=relay_deficits,reward_protocol=reward_protocol,behavior_sha256=collector.policy_sha256,
                 behavior_checkpoint=f'behavior_batch_{len(updates):03d}.pth')
        history.append(row);print('PPO_ROLLOUT '+json.dumps(row),flush=True)
        (out/'progress.json').write_text(json.dumps(history,indent=2))
        if collector.ready or episode==len(args.seeds)-1:
            train=collector.transitions
            old_values=torch.tensor([tr['value'] for tr in train],device='cuda')
            targets=torch.tensor([tr['return'] for tr in train],device='cuda')
            advantages=(targets-old_values).detach()
            roles=sorted({tr['role'] for tr in train});role_stats={}
            for role in roles:
                indices=[i for i,tr in enumerate(train) if tr['role']==role]
                values=advantages[indices]
                role_stats[role]=dict(count=len(indices),advantage_mean=float(values.mean()),
                                     advantage_std=float(values.std(unbiased=False)))
            losses=[];epochs=[]
            for epoch in range(args.ppo_epochs):
                optimizer.zero_grad(set_to_none=True);loss=0.;policy_terms={role:[] for role in roles}
                for i,tr in enumerate(train):
                    output=model(**tr['batch']);dist=torch.distributions.Categorical(logits=output['flat_logits'][0]/args.temperature)
                    newlog=dist.log_prob(torch.tensor(tr['action'],device='cuda'))
                    policy=clipped_policy_loss(newlog,torch.tensor(tr['old_log_prob'],device='cuda'),advantages[i],.2)
                    policy_terms[tr['role']].append(policy)
                    value=.25*(output['value'][0]-targets[i]).square()
                    anchor=args.anchor_weight*torch.nn.functional.cross_entropy(output['flat_logits'],torch.tensor([tr['teacher']],device='cuda'))
                    loss=loss+policy+value+anchor-args.entropy_weight*dist.entropy()
                loss=loss/len(train)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                if epoch==0:
                    actor_parameters=tuple(model.ahgnn.parameters())+tuple(model.actor.parameters())
                    for role,terms in policy_terms.items():
                        gradients=torch.autograd.grad(torch.stack(terms).mean(),actor_parameters,retain_graph=True,allow_unused=True)
                        role_norm=float(torch.sqrt(sum(g.square().sum() for g in gradients if g is not None)))
                        role_stats[role]['role_mean_policy_gradient_norm']=role_norm
                        role_stats[role]['batch_weighted_policy_gradient_norm']=role_norm*len(terms)/len(train)
                loss.backward();gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));optimizer.step()
                losses.append(float(loss.detach()))
                with torch.no_grad():
                    logratios=[]
                    for tr in train:
                        output=model(**tr['batch']);dist=torch.distributions.Categorical(logits=output['flat_logits'][0]/args.temperature)
                        logratios.append(dist.log_prob(torch.tensor(tr['action'],device='cuda'))-tr['old_log_prob'])
                    logratio=torch.stack(logratios);ratio=logratio.exp()
                    kl=float((ratio-1-logratio).mean());clipped=float(((ratio-1).abs()>.2).float().mean())
                epochs.append(dict(epoch=epoch,approx_kl=kl,clip_fraction=clipped,preclip_gradient_norm=gradnorm))
                if kl>args.target_kl:break
            update=dict(batch=len(updates),last_episode=episode,episodes=collector.episodes,transitions=len(train),
                behavior_sha256=collector.policy_sha256,role_stats=role_stats,losses=losses,epochs=epochs)
            updates.append(update);print('PPO_BATCH_UPDATE '+json.dumps(update),flush=True)
            torch.save(dict(protocol_version=PROTOCOL,training_stage='online PPO',model=model.state_dict(),
                optimizer=optimizer.state_dict(),episode=episode,reward_protocol=reward_protocol,initial_sha256=initial_hash,training_args=vars(args)),
                out/f"checkpoint_batch_{update['batch']:03d}.pth")
            (out/'updates.json').write_text(json.dumps(updates,indent=2))
            collector.clear()
        assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest for rel,digest in sources.items())
    assert hashlib.sha256(initial.read_bytes()).hexdigest()==initial_hash
    assert hashlib.sha256(bench.CHECKPOINT.read_bytes()).hexdigest()==frozen_hash
    (out/'complete.json').write_text(json.dumps(dict(episodes=len(history),updates=len(updates),initial_unchanged=True,acceptance_claim=False),indent=2))

if __name__=='__main__':main()
