"""Paired natural rollout of an independent trained candidate.
Same physical runtime, profiles, legal masks and rule baseline as the frozen test.
Experimental model is installed in this process only, after standard setup.
"""
from pathlib import Path
import argparse,json,hashlib,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from extended_gppo_heading import HEADING_PROTOCOL as PROTOCOL,HeadingObservations,heading_batch
from integrations.gppo.gppo_model import GPPOActorCritic
from integrations.gppo.adapter import GPPOAssignment
from gppo_training_records import actor_input_record
from gppo_acceptance_protocol import evaluation_stage
from gppo_motion_capture import capture_motion_frame,append_motion_frame
from gppo_benchmark_domains import load_domain,scenario_paths
from completed_task_stationkeeping import stationkeep_completed_tasks
from integrations.gppo.scheduler import GPPOTaskScheduler

def install_model(adapter,path):
    candidate=torch.load(path,map_location=adapter.device,weights_only=False)
    if candidate.get('protocol_version')!=PROTOCOL:raise ValueError('Wrong experimental checkpoint protocol')
    adapter.model=GPPOActorCritic(edge_feature_dim=6).to(adapter.device)
    adapter.model.load_state_dict(candidate['model'],strict=True)
    adapter.model.eval()
    return candidate

def main():
    import benchmark_fair_search as bench
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--uavs',type=int,choices=[30,60],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=80);p.add_argument('--dynamic',action='store_true');p.add_argument('--scenario');p.add_argument('--warning-rate-mps',type=float,default=1.5);args=p.parse_args()
    evaluation_stage(args.seed,args.uavs,args.steps,False,'dynamic' if args.dynamic else 'static')
    candidate=Path(args.checkpoint).resolve();candidate_hash=hashlib.sha256(candidate.read_bytes()).hexdigest()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    base=bench.MARVELPolicyAdapter
    base_loader=bench.load_and_validate_scenario
    base_runtime=bench.SimulationRuntime
    dynamic_rows=[]
    def dynamic_loader(file):
        return load_domain(base_loader,file,args.scenario,args.dynamic,args.warning_rate_mps)
    class DynamicRuntime(base_runtime):
        def step(self,actions):
            obs,info=super().step(actions)
            scheduler=getattr(self,"high_level_scheduler",None)
            if scheduler is not None and self.current_step%self._mission_step_interval()==0:
                dynamic_rows.append(dict(step=self.current_step,safety_active=sorted(scheduler.safety.active_uav_ids),
                    safety_records=len(scheduler.safety.records),escapes=scheduler.safety.successful_escapes,
                    no_fly_violations=scheduler.safety.no_fly_violations,
                    physical_positions_oracle_only={r.robot_id:r.position.tolist() for r in self.robots},
                    public_delayed_positions={uid:xy.tolist() for uid,xy in scheduler.tracker.positions(scheduler.profile.stale_age_fraction).items()},
                    relay_assignments={target:dict(helper=uid,anchor=anchor.tolist()) for target,(uid,anchor) in scheduler.relay_assignments.items()},
                    task_summary=self.tasks.summary(),relay_release_count=scheduler.relay_release_gate.release_count))
            return obs,info
    bench.load_and_validate_scenario=dynamic_loader
    bench.SimulationRuntime=DynamicRuntime
    current_records=None
    current_motion_file=None
    class CandidateAdapter(base):
        def get_actions(self,observations):
            self.heading_observations.observe(observations)
            current_records['public_headings'].append(dict(time=self.runtime.current_step*self.runtime.dt,
                received=self.heading_observations.current.copy(),
                delayed={uid:self.heading_observations.heading(uid) for uid in self.heading_observations.current}))
            actions=super().get_actions(observations)
            actions=stationkeep_completed_tasks(self.scheduler,actions,
                ordinary_controller=getattr(getattr(self,'_observed_belief',None),'evidence',None))
            append_motion_frame(current_motion_file,capture_motion_frame(self.runtime,self.scheduler,observations,actions))
            return actions
        def setup(self):
            result=super().setup()
            inference=self.scheduler.allocator.adapter
            install_model(inference,candidate)
            self.heading_observations=HeadingObservations(self.scheduler.profile.stale_age_fraction)
            @torch.no_grad()
            def assign(graph,deterministic=True):
                if not graph.num_uavs or not graph.num_subtasks:return None
                batch=heading_batch(graph,self.heading_observations,inference.device)
                output=inference.model(**batch)
                if not bool(output['has_valid_action'][0]):return None
                if deterministic:action=int(output['flat_logits'][0].argmax())
                else:action=int(torch.distributions.Categorical(logits=output['flat_logits'][0]).sample())
                t,u=divmod(action,graph.num_uavs)
                current_records['actor_inputs'].append(actor_input_record(batch,time=graph.current_time,action=action,
                    uav_ids=[u.uav_id for u in graph.uav_states],task_ids=[t.subtask_id for t in graph.subtasks],
                    value=float(output['value'][0])))
                return GPPOAssignment(action,t,u,int(graph.subtasks[t].subtask_id),int(graph.uav_states[u].uav_id),float(output['value'][0]))
            inference.assign=assign
            return result
    bench.MARVELPolicyAdapter=CandidateAdapter
    paths=[ROOT/'scripts/evaluate_extended_gppo_joint_stationkeeping.py',ROOT/'scripts/completed_task_stationkeeping.py',ROOT/'scripts/extended_gppo_heading.py',ROOT/'scripts/gppo_training_records.py',ROOT/'scripts/gppo_motion_capture.py',ROOT/'scripts/benchmark_fair_search.py',
           ROOT/'scripts/train_extended_gppo.py',ROOT/'scripts/gppo_acceptance_protocol.py']
    paths += [ROOT/'scripts/gppo_benchmark_domains.py']+scenario_paths(args.scenario)
    # Snapshot production Python sources too; both arms must use identical code.
    paths+=list((ROOT/'integrations').rglob('*.py'))+list((ROOT/'utils').rglob('*.py'))
    paths += [ROOT/rel for rel in ('parameter.py','test_parameter.py','scripts/filter_scheduler_candidates.py',
        'configs/scenarios/extended_live_gppo.yaml','configs/scenarios/assets/extended_live_gppo.png',
        'configs/dynamics/kinematic_simple.yaml','configs/sensors/ideal.yaml','configs/communication/ideal.yaml',
        'load_model/MARVEL/checkpoint.pth')]
    hashes={str(file.relative_to(ROOT)):hashlib.sha256(file.read_bytes()).hexdigest() for file in paths}
    for file in paths:
        if file.suffix in ('.py','.yaml','.png'):
            saved=out/'source_snapshot'/file.relative_to(ROOT);saved.parent.mkdir(parents=True,exist_ok=True)
            saved.write_bytes(file.read_bytes())
    frozen_hash=hashlib.sha256(bench.CHECKPOINT.read_bytes()).hexdigest()
    manifest=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seed=args.seed,uavs=args.uavs,checkpoint=str(candidate),checkpoint_sha256=candidate_hash,
        protocol=PROTOCOL,frozen_checkpoint_sha256=frozen_hash,source_hashes=hashes,
        comparison='GPPO trained heading-aware actor; self-sensor headings available to both arms, rule uses priority/time; runtime/profile/masks shared',
        controller_variant='postcertification_completed_required_tasks_stationkeeping_experiment',
        stage='natural full-window joint lifecycle development, not final 15% acceptance',horizon_seconds=args.steps,
        domain='dynamic' if args.dynamic else 'static',scenario_override=args.scenario,warning_rate_mps=args.warning_rate_mps if args.dynamic else None)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        results=[]
        for arm in ('gppo','rule'):
            current_records={'public_headings':[],'actor_inputs':[]}
            dynamic_rows.clear()
            current_motion_file=out/arm/'mission_frames.jsonl'
            results.append(bench.run(out,arm,args.uavs,args.seed,args.steps,physical_coverage=True,stop_on_completion=False))
            summary=results[-1]
            summary['search_comparison_valid_before_full_window_check']=summary['valid_for_comparison']
            summary['full_window_complete']=summary['mission_steps']==args.steps
            summary['valid_for_comparison']=bool(summary['valid_for_comparison'] and summary['full_window_complete'])
            (out/arm/'summary.json').write_text(json.dumps(summary,indent=2))
            (out/arm/'joint_lifecycle_trace.json').write_text(json.dumps(dynamic_rows,indent=2))
            (out/arm/'actor_inputs.json').write_text(json.dumps(current_records['actor_inputs'],indent=2))
            (out/arm/'public_headings.json').write_text(json.dumps(current_records['public_headings'],indent=2))
    finally:
        bench.MARVELPolicyAdapter=base
        bench.load_and_validate_scenario=base_loader
        bench.SimulationRuntime=base_runtime
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==candidate_hash
    assert hashlib.sha256(bench.CHECKPOINT.read_bytes()).hexdigest()==frozen_hash
    assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest for rel,digest in hashes.items())
    assert json.loads((out/'gppo/initial.json').read_text())==json.loads((out/'rule/initial.json').read_text())
    (out/'paired.json').write_text(json.dumps(results,indent=2))
    gm=results[0]['metric']['mean_capped_seconds'];rm=results[1]['metric']['mean_capped_seconds']
    (out/'comparison.json').write_text(json.dumps(dict(valid=all(r['valid_for_comparison'] for r in results),
        efficiency_gain=rm/gm-1 if gm else None,acceptance_claim=False),indent=2))

if __name__=='__main__':main()
