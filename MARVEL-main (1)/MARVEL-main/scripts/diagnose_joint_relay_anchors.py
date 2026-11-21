"""Offline Relay anchor diagnosis: public telemetry and observed cells; map only oracle."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.mesh_relay import mesh_components
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.obstacle_manager import ObstacleManager

def main():
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True);p.add_argument('--arm',default='gppo');p.add_argument('--out',required=True);args=p.parse_args()
    folder=ROOT/args.pair/args.arm
    frames=[json.loads(s) for s in (folder/'mission_frames.jsonl').read_text().splitlines()]
    traces=json.loads((folder/'joint_lifecycle_trace.json').read_text());frame=frames[-1]
    trace=next(t for t in traces if t['step']==frame['step'])
    config=json.loads((folder/'config.json').read_text());ob=ObstacleManager(config['environment']);ob.load_from_file(config['environment']['map_file'])
    rec=ObservedSearchRecovery(ob.frame);rec.cells={(c,r):v for c,r,v in frame['observed_recovery']['cells']}
    public={int(uid):np.array(xy) for uid,xy in trace['public_delayed_positions'].items()}
    assignments=frame['public_relay_assignments'];helpers={a['helper'] for a in assignments.values()};groups=mesh_components(public,20,helpers)
    known=[np.array([c+.5,r+.5]) for (c,r),label in rec.cells.items() if label==0 and rec.segment_safe([c+.5,r+.5],[c+.5,r+.5])]
    results=[]
    for target,assignment in assignments.items():
        group=next((g for g in groups if int(target) in g),None)
        if group is None:continue
        others=[g for g in groups if g!=group];core=min(others,key=lambda g:(-len(g),g))
        gap,t,a=min((float(np.linalg.norm(public[t]-public[a])),t,a) for t in group for a in core)
        anchor=np.array(assignment['anchor']);cell=rec.frame.world_to_cell_floor(anchor)
        feasible=[xy for xy in known if np.linalg.norm(xy-public[t])<=20 and np.linalg.norm(xy-public[a])<=20]
        feasible.sort(key=lambda xy:(float(np.linalg.norm(xy-anchor)),float(xy[0]),float(xy[1])))
        any_boundary=[xy for xy in known if min(np.linalg.norm(xy-public[u]) for u in group)<=20 and min(np.linalg.norm(xy-public[u]) for u in core)<=20]
        selected=feasible[0] if feasible else None
        results.append(dict(target=int(target),helper=assignment['helper'],anchor=anchor.tolist(),
            anchor_observed_label=rec.cells.get(tuple(cell)),anchor_oracle_collision=ob.check_collision(anchor,radius=.2),
            boundary_pair=[t,a],gap=gap,known_free_radio_lens_candidates=len(feasible),
            any_boundary_known_free_radio_lens_candidates=len(any_boundary),
            nearest_observed_free_candidate=None if selected is None else selected.tolist(),
            candidate_oracle_collision=None if selected is None else ob.check_collision(selected,radius=.2),
            path_feasibility='not checked',deployment='candidate uses observed cells and public telemetry only'))
    result=dict(frame_step=frame['step'],public_component_sizes=[len(g) for g in mesh_components(public,20)],
        nonhelper_component_sizes=[len(g) for g in groups],anchors=results,acceptance_claim=False)
    (ROOT/args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
