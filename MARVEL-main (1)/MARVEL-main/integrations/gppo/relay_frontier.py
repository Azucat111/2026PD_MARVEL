"""Extended observed-only information gathering; never a truth-map planner."""
from collections import deque
import numpy as np
from utils.sensor_models import IdealSensor
from integrations.gppo.mesh_relay import mesh_components

class ObservedRelayFrontier:
    def __init__(self,recovery,*,comm_range=20.,sensor_range=10.,fov=120.,max_frontiers=128):
        self.recovery=recovery;self.comm_range=float(comm_range)
        self.sensor=IdealSensor(dict(range=float(sensor_range),fov=float(fov),los_occlusion=True),frame=recovery.frame)
        self.max_frontiers=int(max_frontiers);self.visited=set();self.routes={}
        if self.comm_range<=0 or self.max_frontiers<=0:raise ValueError('Positive search bounds required')

    def select(self,positions,*,reserved=(),segment_filter=None):
        rec=self.recovery;frame=rec.frame
        points={int(u):np.asarray(p,dtype=float) for u,p in positions.items()}
        if any(p.shape!=(2,) or not np.isfinite(p).all() for p in points.values()):raise ValueError('Finite public positions required')
        groups=mesh_components(points,self.comm_range)
        if len(groups)<2 or frame.width_cells*frame.height_cells>16384:return None
        excluded=set(map(int,reserved));free={c for c,v in rec.cells.items() if v==0};baseline=len(groups[0])
        def centre(c):return np.asarray(frame.origin)+(np.asarray(c,dtype=float)+.5)*frame.cell_size
        frontier={c for c in free if c not in self.visited and any(frame.contains_cell(c[0]+dx,c[1]+dy) and (c[0]+dx,c[1]+dy) not in rec.cells for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)])}
        if not frontier:return None
        known=np.full(frame.shape,127,dtype=np.uint8)
        for (c,r),v in rec.cells.items():
            if frame.contains_cell(c,r):known[r,c]=v
        core=groups[0];remote=[u for g in groups[1:] for u in g]
        # Prefer a disconnected boundary robot; other helpers remain a fallback.
        order=sorted(remote,key=lambda u:(min(np.linalg.norm(points[u]-points[v]) for v in core),u))+list(core)
        for uid in order:
            if uid in excluded or uid not in rec.observed_positions:continue
            start=np.asarray(rec.observed_positions[uid],dtype=float)
            if start.shape!=(2,) or not np.isfinite(start).all():continue
            targets=[points[u] for u in (core if uid in remote else remote)]
            clear=self._clear_for(uid,start,segment_filter)
            previous=self._known_free_paths(start,free,centre,clear)
            candidates=[]
            ordered=sorted(frontier & previous.keys(),key=lambda c:(min(np.linalg.norm(centre(c)-t) for t in targets),c))[:self.max_frontiers]
            for cell in ordered:
                anchor=centre(cell);future=mesh_components({**points,uid:anchor},self.comm_range)
                if len(future[0])<baseline or not all(any(set(g)-{uid}<=set(q) for q in future) for g in groups):continue
                target=min(targets,key=lambda t:np.linalg.norm(anchor-t));direction=target-anchor
                heading=float(np.degrees(np.arctan2(direction[1],direction[0]))%360)
                potential=sum(known[r,c]==127 for c,r in self.sensor._visible_cells(anchor,heading,known))
                if not potential:continue
                chain=[];node=cell
                while node is not None:chain.append(node);node=previous[node]
                vertices=[start,*[centre(c) for c in reversed(chain)]]
                length=sum(float(np.linalg.norm(a-b)) for a,b in zip(vertices,vertices[1:]));gap=float(np.linalg.norm(anchor-target))
                candidates.append((gap+.05*length-.01*potential,cell,dict(uid=uid,anchor=anchor,heading=heading,path=vertices,forecast_lcc=len(future[0]),potential_unknown_cells=int(potential),known_path_m=length)))
            if candidates:
                best=min(candidates,key=lambda q:q[:2]);self.visited.add(best[1]);return best[2]
        return None

    def _clear_for(self,uid,start,segment_filter):
        rec=self.recovery;peers=rec.nearby_positions.get(uid,[])
        def clear(a,b):
            a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
            if not rec.segment_safe(a,b) or (segment_filter is not None and not segment_filter(a,b)):return False
            delta=b-a;length2=float(delta@delta)
            for peer in peers:
                t=float(np.clip(np.dot(peer-a,delta)/length2,0.,1.)) if length2>1e-20 else 0.
                if np.linalg.norm(a+t*delta-peer)<min(.5,float(np.linalg.norm(start-peer)))-1e-9:return False
            return True
        return clear

    def _known_free_paths(self,start,free,centre,clear):
        # Same bounded four-neighbour observed FREE traversal used by select.
        c0=self.recovery.frame.world_to_cell_floor(start)
        starts=[(c0[0]+dx,c0[1]+dy) for dx in [-1,0,1] for dy in [-1,0,1] if (c0[0]+dx,c0[1]+dy) in free and clear(start,centre((c0[0]+dx,c0[1]+dy)))]
        previous={c:None for c in starts};queue=deque(starts)
        while queue:
            cell=queue.popleft()
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                other=(cell[0]+dx,cell[1]+dy)
                if other in free and other not in previous and clear(centre(cell),centre(other)):
                    previous[other]=cell;queue.append(other)
        return previous

    def known_route(self,uid,goal,*,segment_filter=None):
        """Reuse observed FREE paths; peer/warning checks remain current.

        No truth map, unobserved grid cell or forecast moving peer is an input.
        The returned path is a present-observation certificate, not a guarantee
        for future hazards. Runtime collision and SafetyShield stay authoritative.
        """
        uid=int(uid);rec=self.recovery;frame=rec.frame;goal=np.asarray(goal,dtype=float)
        start=rec.observed_positions.get(uid)
        if start is None:self.routes.pop(uid,None);return None
        start=np.asarray(start,dtype=float)
        if start.shape!=(2,) or goal.shape!=(2,) or not np.isfinite(start).all() or not np.isfinite(goal).all():raise ValueError('Finite own pose and public goal required')
        clear=self._clear_for(uid,start,segment_filter)
        failed=[target for target,origin in rec.failed.get(uid,[]) if np.linalg.norm(start-origin)<=.5]
        def usable(vertices):
            return len(vertices)>1 and not any(np.linalg.norm(vertices[1]-target)<1e-6 for target in failed) and all(clear(a,b) for a,b in zip(vertices,vertices[1:]))
        cached=self.routes.get(uid)
        if cached is not None and np.array_equal(cached['goal'],goal):
            remaining=[p.copy() for p in cached['points']]
            while remaining and np.linalg.norm(remaining[0]-start)<=.1:remaining.pop(0)
            vertices=[start,*remaining]
            if usable(vertices):self.routes[uid]['points']=remaining;return [p.copy() for p in vertices]
        self.routes.pop(uid,None)
        direct=[start,goal]
        if usable(direct):self.routes[uid]=dict(goal=goal.copy(),points=[goal.copy()]);return [p.copy() for p in direct]
        if frame.width_cells*frame.height_cells>16384:return None
        free={c for c,v in rec.cells.items() if v==0}
        def centre(c):return np.asarray(frame.origin)+(np.asarray(c,dtype=float)+.5)*frame.cell_size
        previous=self._known_free_paths(start,free,centre,clear);candidates=[]
        for cell in previous:
            endpoint=centre(cell)
            if np.linalg.norm(endpoint-goal)>1.5 or not clear(endpoint,goal):continue
            chain=[];node=cell
            while node is not None:chain.append(centre(node));node=previous[node]
            vertices=[start,*reversed(chain),goal]
            while len(vertices)>2 and np.linalg.norm(vertices[1]-start)<=.1:vertices.pop(1)
            if not usable(vertices):continue
            length=sum(float(np.linalg.norm(a-b)) for a,b in zip(vertices,vertices[1:]))
            candidates.append((length,cell,vertices))
        if not candidates:return None
        vertices=min(candidates,key=lambda c:c[:2])[2]
        self.routes[uid]=dict(goal=goal.copy(),points=[p.copy() for p in vertices[1:]])
        return [p.copy() for p in vertices]
