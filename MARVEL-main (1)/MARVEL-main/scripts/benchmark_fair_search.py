"""Bounded natural-from-reset common-condition scheduling comparison.
Only allocator inference changes. Sensor/motion/lifecycle/capacity are shared.
Smoke experiments are explicitly not the final >=15% acceptance evidence.
"""
from pathlib import Path
import sys,json,time,csv,hashlib,argparse,subprocess,random,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from copy import copy
from filter_scheduler_candidates import limit_travel_candidates
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch,logging
torch.set_num_threads(1)
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime
from utils.policy_adapter import MARVELPolicyAdapter
from integrations.gppo.checkpoint import load_frozen_gppo
from integrations.gppo.config_profile import prepare_gppo_config
from integrations.rule_scheduling import choose_rule_assignment
from gppo_benchmark_domains import validate_physical_warning_rate
from integrations.search_efficiency import detection_efficiency,observed_free_coverage
CHECKPOINT=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')

def run(out,arm,uavs,seed,steps,candidate_budget=0,physical_coverage=False,stop_on_completion=False,wall_budget_seconds=600.):
    wall_budget_seconds=float(wall_budget_seconds)
    if not np.isfinite(wall_budget_seconds) or wall_budget_seconds<=0:
        raise ValueError('wall_budget_seconds must be finite and positive')
    config=load_and_validate_scenario(ROOT/'configs/scenarios/extended_live_gppo.yaml')
    config['robots']=[{'id_range':[0,uavs-1],'type':'explorer','team_id':1,'config':{
        'fov':120,'sensor_range':10.,'velocity':1.,'yaw_rate':35,'initial_positions':'random_safe'}}]
    config['scenario']['random_seed']=seed
    config['task_scheduler']={'mode':'gppo','checkpoint':str(CHECKPOINT),
        'base_position':[0.,0.],'search_scenario_seed':seed}
    _,checkpoint=load_frozen_gppo(str(CHECKPOINT),device='cpu')
    config=prepare_gppo_config(config,checkpoint)
    validate_physical_warning_rate(config)
    if physical_coverage:
        for task in config['tasks']:
            if task['type']=='exploration':task['params']['coverage_basis']='observed_free'
    config['policy']={'device':'cuda','batch_inference':True}
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    runtime=SimulationRuntime(config)
    obs=runtime.reset()
    adapter=MARVELPolicyAdapter(runtime);adapter.setup()
    assert adapter._using_policy, "MARVEL checkpoint did not load; comparison invalid"
    allocator=adapter.scheduler.allocator
    original=allocator.adapter.assign
    decisions=[]
    def assign(graph,deterministic=True):
        original_mask=graph.action_mask.copy()
        if candidate_budget:
            graph=copy(graph)
            graph.action_mask=limit_travel_candidates(graph,candidate_budget)
        choice=(choose_rule_assignment(graph,deterministic) if arm=='rule' else original(graph,deterministic=deterministic))
        decisions.append({'step':runtime.current_step,'uav_ids':[u.uav_id for u in graph.uav_states],
            'task_ids':[t.subtask_id for t in graph.subtasks],
            'route_support_masks':getattr(allocator,'last_route_support_masks',[]),
            'deadline_masks':getattr(allocator,'last_deadline_masks',[]),
            'observed_reachability_masks':getattr(allocator,'last_observed_reachability_masks',[]),
            'original_mask':original_mask.tolist(),'candidate_budget':candidate_budget,
            'mask':graph.action_mask.tolist(),'observed_total_time':graph.edge_features[:,:,2].tolist(),
            'route_estimates':[{'task_id':tid,'uav_id':uid,**details} for (tid,uid),details in getattr(allocator,'last_travel_estimates',{}).items()],
            'current_time':graph.current_time,'raw_uav_features':graph.uav_features.tolist(),
            'raw_task_features':graph.task_features.tolist(),'raw_edge_features':graph.edge_features.tolist(),
            'model_uav_features':graph.model_uav_features.tolist(),'model_task_features':graph.model_task_features.tolist(),
            'model_edge_features':graph.model_edge_features.tolist(),
            'capability_matrix':graph.capability_matrix.tolist(),'precedence_matrix':graph.precedence_matrix.tolist(),
            'chosen':None if choice is None else {'uav_id':choice.uav_id,'subtask_id':choice.subtask_id}})
        return choice
    allocator.adapter.assign=assign
    state={'initial_positions':[r.position.tolist() for r in runtime.robots],
        'public_heats':[{'id':h.heat_id,'position':np.asarray(h.position).tolist()} for h in adapter.scheduler.heat_points]}
    folder=out/arm;folder.mkdir()
    (folder/'config.json').write_text(json.dumps(config,indent=2,default=str))
    (folder/'initial.json').write_text(json.dumps(state,indent=2))
    policy_failures=[]
    policy_logger=logging.getLogger('utils.policy_adapter')
    previous_level=policy_logger.level
    class PolicyFailureRecorder(logging.Handler):
        def emit(self,record):
            message=record.getMessage()
            if any(marker in message for marker in ('update_graph failed','update_planning_state failed','Action selection failed')):
                policy_failures.append({'step':runtime.current_step,'message':message})
    failure_recorder=PolicyFailureRecorder()
    policy_logger.addHandler(failure_recorder);policy_logger.setLevel(logging.DEBUG)
    interval=runtime._mission_step_interval()
    started=time.monotonic(); collisions=[];rows=[]
    try:
        with (folder/'progress.csv').open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['mission','coverage','active','serviced','found','search','relay','collisions'])
            writer.writeheader()
            for mission in range(1,steps+1):
                actions=adapter.get_actions(obs)
                for tick in range(interval):
                    obs,info=runtime.step(actions)
                    collisions.extend(info['collisions'])
                    if stop_on_completion and runtime.tasks.all_complete():break
                scheduler=adapter.scheduler
                row={'mission':mission,'coverage':runtime.exploration_rate,'active':scheduler.search_activated,
                    'serviced':len(scheduler.serviced_heat_ids),'found':sum(e['type']=='target_found' for e in runtime.events),
                    'search':len(scheduler.search_assignments),'relay':len(scheduler.relay_assignments),'collisions':len(collisions)}
                rows.append(row);writer.writerow(row);handle.flush()
                if mission%10==0:print(arm,mission,row,round(time.monotonic()-started,1),flush=True)
                if (stop_on_completion and runtime.tasks.all_complete()) or time.monotonic()-started>wall_budget_seconds:break
    finally:
        allocator.adapter.assign=original
        policy_logger.removeHandler(failure_recorder);policy_logger.setLevel(previous_level)
    target_count=runtime.search_scenario.target_count
    horizon=steps*interval*runtime.dt
    valid=(runtime.current_step>=steps*interval or runtime.tasks.all_complete()) and not policy_failures
    summary={'arm':arm,'uavs':uavs,'seed':seed,'mission_steps':mission,'natural_from_reset':True,
        'smoke_only':steps<300,'valid_for_comparison':valid,'required_nonrelay_tasks_complete':runtime.tasks.all_complete(),
        'elapsed_simulation_seconds':runtime.current_step*runtime.dt,'task_summary':runtime.tasks.summary(),
        'candidate_budget':candidate_budget,'physical_coverage':physical_coverage,'metric':detection_efficiency(runtime.events,target_count,runtime.dt,horizon),
        'physical_free_coverage':observed_free_coverage(runtime.explored_cells,runtime.obstacles.get_occupancy_grid()),
        'activation_step':adapter.scheduler.search_activation_step,'serviced':sorted(adapter.scheduler.serviced_heat_ids),
        'policy_execution_failures':policy_failures,
        'collision_types':{kind:sum(e['type']==kind for e in collisions) for kind in {e['type'] for e in collisions}},
        'wall_budget_seconds':wall_budget_seconds,'seconds':time.monotonic()-started}
    (folder/'collisions.json').write_text(json.dumps(collisions,indent=2))
    (folder/'events.json').write_text(json.dumps(runtime.events,indent=2,default=str))
    (folder/'decisions.json').write_text(json.dumps(decisions,indent=2))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--uavs',type=int,default=30)
    parser.add_argument('--seed',type=int,default=12345);parser.add_argument('--steps',type=int,default=80)
    parser.add_argument('--candidate-budget',type=int,default=0)
    parser.add_argument('--physical-coverage',action='store_true')
    parser.add_argument('--stop-on-completion',action='store_true')
    parser.add_argument('--wall-budget-seconds',type=float,default=600.)
    args=parser.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    files=['utils/node_manager.py','utils/utils.py','integrations/observed_node_lattice.py','integrations/gppo/gppo_model.py','integrations/gppo/task_graph.py','utils/policy_adapter.py','utils/simulation_runtime.py','integrations/gppo/scheduler.py',
        'integrations/gppo/search_recovery.py','integrations/gppo/search_deadline.py','integrations/gppo/route_support.py','integrations/gppo/observed_reachability.py','integrations/gppo/extended_travel.py','integrations/gppo/event_allocator.py','integrations/observed_belief.py','integrations/rule_scheduling.py',
        'integrations/search_efficiency.py','utils/task_manager.py','scripts/filter_scheduler_candidates.py','utils/safety_shield.py','scripts/benchmark_fair_search.py','configs/scenarios/extended_live_gppo.yaml','configs/scenarios/assets/extended_live_gppo.png']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_hashes':hashes,'marvel_checkpoint_hash':hashlib.sha256((ROOT/'load_model/MARVEL/checkpoint.pth').read_bytes()).hexdigest(),'gppo_checkpoint_hash':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'comparison':'only allocator inference changes; same optional candidate budget in both arms',
        'candidate_budget':args.candidate_budget,'physical_coverage':args.physical_coverage,
        'all_rngs_seeded':True,'torch_deterministic':True,'python_hash_seed':os.environ.get('PYTHONHASHSEED'),
        'requested_horizon_seconds':args.steps,'wall_budget_seconds':args.wall_budget_seconds,'smoke_only':args.steps<300}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    results=[run(out,arm,args.uavs,args.seed,args.steps,args.candidate_budget,args.physical_coverage,args.stop_on_completion,wall_budget_seconds=args.wall_budget_seconds) for arm in ('gppo','rule')]
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert json.loads((out/'gppo/initial.json').read_text())==json.loads((out/'rule/initial.json').read_text())
    (out/'paired.json').write_text(json.dumps(results,indent=2))
