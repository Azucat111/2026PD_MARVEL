"""Offline comparison of frozen BFS and existing real-pose certified graph paths.
Uses captured observed cells/graphs and own sensor poses only. No truth map.
"""
from pathlib import Path
import sys,argparse,json,heapq
import numpy as np
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.observed_graph import observed_next_hop

def diagnose(f,uid,goal):
    meta=f['observed_recovery']['frame']
    rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']))
    rec.cells={(c,r):v for c,r,v in f['observed_recovery']['cells']}
    own=f['selected_own_observations'].get(str(uid))
    if own is None:return {'step':f['step'],'uid':uid,'missing':'own captured observation'}
    position=np.array(own['position']);g=f['observed_graphs_diagnostic_only'][str(uid)]
    coords=np.array(g['node_coords']);adj=np.ones((len(coords),len(coords)))
    for u,v in g['zero_adjacency_edges']:adj[u,v]=0
    agent=SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=g['neighbor_indices'])
    raw=observed_next_hop(agent,goal)
    cache={}
    def clear(a,b):
        key=tuple(sorted((tuple(a),tuple(b))))
        if key not in cache:cache[key]=rec.segment_safe(a,b)
        return cache[key]
    dist=np.full(len(coords),np.inf);prev=np.full(len(coords),-1,dtype=int);heap=[]
    for i in np.unique(np.append(g['neighbor_indices'],g['current_index'])):
        cost=float(np.linalg.norm(position-coords[i]))
        if cost<=6 and clear(position,coords[i]):dist[i]=cost;heapq.heappush(heap,(cost,int(i)))
    while heap:
        cost,u=heapq.heappop(heap)
        if cost!=dist[u]:continue
        for v in np.flatnonzero(adj[u]==0):
            if v==u or not clear(coords[u],coords[v]):continue
            alternative=cost+float(np.linalg.norm(coords[u]-coords[v]))
            if alternative<dist[v]:dist[v]=alternative;prev[v]=u;heapq.heappush(heap,(alternative,int(v)))
    reachable=np.flatnonzero(np.isfinite(dist));path=[]
    if len(reachable):
        destination=int(reachable[np.argmin(np.linalg.norm(coords[reachable]-goal,axis=1))]);v=destination
        while v>=0:path.append(int(v));v=int(prev[v])
        path.reverse()
        while len(path)>1 and np.linalg.norm(coords[path[0]]-position)<=.1:path.pop(0)
    command=np.array(f['actual_commands'][str(uid)]['waypoint'])
    return dict(step=f['step'],uid=uid,own_position=position.tolist(),goal=goal.tolist(),
        raw_bfs_waypoint=None if raw is None else raw[0].tolist(),raw_bfs_prefix_known_free=None if raw is None else rec.segment_safe(position,raw[0]),
        actual_command=command.tolist(),actual_prefix_known_free=rec.segment_safe(position,command),
        certified_static_path=[position.tolist()]+[coords[i].tolist() for i in path],
        certified_first=None if not path else coords[path[0]].tolist(),
        certificate='all captured FREE five-probe segments only; future peer traffic not predicted',production_unchanged=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',nargs='+',type=int);a=p.parse_args();rows=[]
    with (ROOT/a.frames).open() as stream:
        for line in stream:
            f=json.loads(line)
            if a.steps is not None and f['step'] not in a.steps:continue
            for task,assignment in f['public_relay_assignments'].items():rows.append(diagnose(f,int(assignment['helper']),np.array(assignment['anchor'])))
    with (ROOT/a.out).open('x') as stream:json.dump(rows,stream,indent=2)
    print(json.dumps(rows))
if __name__=='__main__':main()
