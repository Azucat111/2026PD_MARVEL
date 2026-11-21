"""Observed background motion after required task completion; no counterfactual claim."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True);p.add_argument('--out',required=True);a=p.parse_args();results={}
    for arm in ('gppo','rule'):
        folder=ROOT/a.pair/arm;trace={t['step']:t for t in json.loads((folder/'joint_lifecycle_trace.json').read_text())}
        frames=[json.loads(line) for line in (folder/'mission_frames.jsonl').read_text().splitlines()]
        total=0.;segments=0;start=None;uids=set()
        for first,last in zip(frames,frames[1:]):
            t=trace.get(first['step'])
            if t is None or not all(v['status']=='complete' for v in t['task_summary'].values() if v['type'] in ('exploration','target_search')):continue
            if start is None:start=first['time']
            used=set(t['safety_active'])|set(trace.get(last['step'],{}).get('safety_active',[]))
            for frame in (first,last):
                used.update(map(int,frame['public_search_assignments'].values()))
                used.update(int(v['helper']) for v in frame['public_relay_assignments'].values())
            for uid,pose in first['physical_poses_diagnostic_only'].items():
                if int(uid) in used:continue
                delta=last['physical_poses_diagnostic_only'][uid]['travel_distance']-pose['travel_distance']
                assert delta>=-1e-8
                if delta>1e-8:total+=delta;segments+=1;uids.add(int(uid))
        results[arm]=dict(required_tasks_complete_from_seconds=start,unassigned_postcompletion_actual_travel_m=total,moving_interval_count=segments,distinct_unassigned_moving_uavs=len(uids),final_relay_ratio=trace[max(trace)]['task_summary']['T3_relay']['progress'])
    result=dict(pair=a.pair,results=results,source='captured actual odometers and recorded task statuses/assignments',interpretation='background movement continues after required tasks are complete; effect of stationkeeping requires a separate same-condition runtime experiment',acceptance_claim=False)
    with (ROOT/a.out).open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
