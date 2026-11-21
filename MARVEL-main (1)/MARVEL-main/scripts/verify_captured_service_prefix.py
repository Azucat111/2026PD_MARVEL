"""Literal captured service pose; observed selection and separate map collision oracle."""
from pathlib import Path
import argparse,json,sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.obstacle_manager import ObstacleManager
from utils.dynamics_models import KinematicSimple

def main():
    p=argparse.ArgumentParser();p.add_argument('--reconstruction',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    source=ROOT/args.reconstruction
    frames=[json.loads(line) for line in (source/'gppo/mission_frames.jsonl').read_text().splitlines()]
    frame=next(f for f in frames if f['step']==30)
    uid=17;heat=1;pose=frame['physical_poses_diagnostic_only'][str(uid)]
    config=json.loads((source/'gppo/config.json').read_text())
    ob=ObstacleManager(config['environment']);ob.load_from_file(config['environment']['map_file'])
    assert ob.frame.describe()==frame['observed_recovery']['frame']
    rec=ObservedSearchRecovery(ob.frame)
    rec.cells={(c,r):label for c,r,label in frame['observed_recovery']['cells']}
    own=frame['selected_own_observations'][str(uid)]
    rec.nearby_positions={uid:[np.array(point) for point in own.get('nearby_robot_positions',{}).values()]}
    start=np.array(pose['position']);anchor=np.array(frame['public_heat_positions'][str(heat)])
    chosen=rec.select(uid,start,anchor,None,anchor,None,max_anchor_distance=3.5)
    assert np.linalg.norm(chosen-start)>.1 and rec.segment_safe(start,chosen)
    original_grid=ob._file_grid;ob._file_grid=np.ones_like(original_grid)
    same=rec.select(uid,start,anchor,None,anchor,None,max_anchor_distance=3.5)
    assert np.array_equal(chosen,same);ob._file_grid=original_grid
    model=KinematicSimple(config['dynamics']['params']);position=start.copy();heading=pose['heading']
    desired=frame['actual_commands'][str(uid)]['heading'];ticks=[]
    for tick in range(10):
        update,_=model.step(position,chosen,heading,desired,dt=.1)
        point=update['position'];blocked,reason=ob.check_collision(point,radius=.2)
        assert not blocked and rec.segment_safe(position,point)
        assert np.linalg.norm(point-anchor)<=3.5+1e-9
        assert all(np.linalg.norm(point-peer)>=.5 for peer in rec.nearby_positions[uid])
        ticks.append(dict(tick=tick+1,position=point.tolist(),heading=update['heading'],collision=reason))
        position=point;heading=update['heading']
    result=dict(uid=uid,heat=heat,original_recorded_step=30,start=start.tolist(),anchor=anchor.tolist(),
        selected_waypoint=chosen.tolist(),desired_heading=desired,ticks=ticks,
        observed_map_only_selection=True,oracle_independent_choice=True,physical_radius=.2,
        statement='Legal observed first prefix only; not proof of target detection or complete service recovery')
    out=ROOT/args.out;out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
