"""Extended-only Search prefix recovery from sensor labels and own feedback.

No obstacle manager or truth-map input belongs to this controller. A certified
segment covers the existing runtime's five grid probes, not a full disk nor
future moving obstacles. Physical rejection and SafetyShield remain authoritative.
"""
from __future__ import annotations

import numpy as np


class ObservedSearchRecovery:
    def __init__(self, frame):
        self.frame = frame
        self.reset_generation = 0
        self.cells = {}
        self.prefixes = {}
        self.prefix_headings = {}
        self.anchors = {}
        self.failed = {}
        self.last_feedback_step = {}
        self.assignment_tokens = {}
        self.sensing_uids = set()
        self.nearby_positions = {}
        self.observed_positions = {}
        self._observed_reachability_gate = None
        self._authoritative_areal_grid = False
        self._observed_reachability_step = None

    def reset_uid(self, uid):
        uid = int(uid)
        self.sensing_uids.discard(uid)
        for state in (self.prefixes, self.prefix_headings, self.anchors, self.failed, self.last_feedback_step):
            state.pop(uid, None)

    def reset(self):
        self.reset_generation += 1
        self._observed_reachability_gate = None
        self._authoritative_areal_grid = False
        self._observed_reachability_step = None
        self.sensing_uids.clear()
        for state in (self.cells, self.prefixes, self.prefix_headings, self.anchors, self.failed,
                      self.last_feedback_step, self.assignment_tokens, self.nearby_positions, self.observed_positions):
            state.clear()

    def sync_assignments(self, assignments):
        tokens = {int(uid): int(heat) for heat, uid in assignments.items()}
        for uid in set(self.assignment_tokens) | set(tokens):
            if self.assignment_tokens.get(uid) != tokens.get(uid):
                self.reset_uid(uid)
        self.assignment_tokens = tokens

    def observe(self, observations):
        self._observed_reachability_step = None
        self.observed_positions.clear()
        for raw_uid, observation in observations.items():
            uid = int(raw_uid)
            position = np.asarray(observation.get("position"), dtype=float)
            if position.shape == (2,) and np.all(np.isfinite(position)):
                self.observed_positions[uid] = position.copy()
            else:
                self.observed_positions.pop(uid, None)
            peers=[]
            for raw_peer,raw_position in observation.get("nearby_robot_positions",{}).items():
                point=np.asarray(raw_position,dtype=float)
                if int(raw_peer)!=uid and point.shape==(2,) and np.all(np.isfinite(point)):
                    peers.append(point.copy())
            self.nearby_positions[uid]=peers
            description = observation.get("occupancy_frame")
            expected = self.frame.describe()
            if description is not None and all(
                description.get(key) == expected[key]
                for key in ("origin", "cell_size", "width_cells", "height_cells")
            ):
                self._authoritative_areal_grid = (
                    observation.get("visible_occupancy_semantics") == "areal_file_grid")
                for col, row, value in observation.get("visible_occupancy", []):
                    col, row, value = int(col), int(row), int(value)
                    if self.frame.contains_cell(col, row) and value in (0, 1):
                        self.cells[col, row] = value
            feedback = observation.get("motion_feedback", {})
            step = feedback.get("step")
            if step is None or step == self.last_feedback_step.get(uid):
                continue
            self.last_feedback_step[uid] = step
            position = np.asarray(observation.get("position"), dtype=float)
            proposal = np.asarray(feedback.get("proposal"), dtype=float)
            if (feedback.get("accepted") is True
                    and feedback.get("grid_probes_clear") is True
                    and position.shape == (2,) and proposal.shape == (2,)
                    and np.array_equal(position, proposal)):
                # An accepted physical collision test proves exactly these
                # current five probe cells free, even behind the sensor FoV.
                # This is own motion evidence, never an unseen map query.
                for offset in ((0, 0), (.2, 0), (-.2, 0), (0, .2), (0, -.2)):
                    cell = self.frame.world_to_cell_floor(position + offset)
                    self.cells[self.frame.clip_cell(*cell)] = 0
            if feedback.get("accepted") is False or feedback.get("shield_rejected") is True:
                target = np.asarray(feedback.get("requested_target", feedback.get("target")), dtype=float)
                if target.shape != (2,) or position.shape != (2,):
                    continue
                entries = self.failed.setdefault(uid, [])
                entries.append((target.copy(), position.copy()))
                self.failed[uid] = entries[-16:]
                self.prefixes.pop(uid, None)
                self.prefix_headings.pop(uid, None)

    def _ray_cells(self, start, end):
        """Enumerate every containing cell along a straight probe trajectory.

        Split at all lattice boundaries, checking endpoints and each interval
        midpoint. This includes arbitrarily short corner-cell traversals.
        """
        delta = end - start
        times = {0., 1.}
        origin = np.asarray(self.frame.origin)
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-15:
                continue
            lo, hi = sorted(((start[axis] - origin[axis]) / self.frame.cell_size,
                             (end[axis] - origin[axis]) / self.frame.cell_size))
            for boundary in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
                t = (origin[axis] + boundary * self.frame.cell_size - start[axis]) / delta[axis]
                if 0. < t < 1.:
                    times.add(float(t))
        times = sorted(times)
        samples = times + [(a + b) / 2 for a, b in zip(times, times[1:])]
        convert = getattr(self.frame, "world_to_cells_floor", None)
        if callable(convert):
            points = start + np.asarray(samples, dtype=float)[:, None] * delta
            for col, row in convert(points, clip=True):
                yield int(col), int(row)
        else:
            for t in samples:
                col, row = self.frame.world_to_cell_floor(start + t * delta)
                yield self.frame.clip_cell(col, row)

    def segment_safe(self, start, end, radius=.2):
        start = np.asarray(start, dtype=float)[:2]
        end = np.asarray(end, dtype=float)[:2]
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            return False
        lower, upper = self.frame.bounds_min, self.frame.bounds_max
        if any(np.any(point < lower) or np.any(point > upper) for point in (start, end)):
            return False
        for offset in ((0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)):
            for cell in self._ray_cells(start + offset, end + offset):
                if self.cells.get(cell) != 0:
                    return False
        return True

    def select(self, uid, position, anchor, agent, preferred, base, base_heading=None,
               max_anchor_distance=None, segment_filter=None, progress_target=None,
               prefer_base=False, fallback_preferred=None):
        uid = int(uid)
        position = np.asarray(position, dtype=float)[:2]
        anchor = np.asarray(anchor, dtype=float)[:2]
        progress = anchor if progress_target is None else np.asarray(progress_target,dtype=float)[:2]
        if prefer_base and base is not None and np.linalg.norm(np.asarray(base,dtype=float)[:2]-position)>.1:
            progress=np.asarray(base,dtype=float)[:2]
        if uid not in self.anchors or not np.array_equal(self.anchors[uid], anchor):
            self.prefixes.pop(uid, None)
            self.prefix_headings.pop(uid, None)
            self.failed.pop(uid, None)
            self.anchors[uid] = anchor.copy()

        entries = [(target, origin) for target, origin in self.failed.get(uid, [])
                   if np.linalg.norm(position - origin) <= .5]
        self.failed[uid] = entries

        def peers_clear(target):
            delta=target-position
            length2=float(np.dot(delta,delta))
            for peer in self.nearby_positions.get(uid,[]):
                t=float(np.clip(np.dot(peer-position,delta)/length2,0.,1.)) if length2>1e-20 else 0.
                closest=float(np.linalg.norm(position+t*delta-peer))
                limit=min(.5,float(np.linalg.norm(position-peer)))
                if closest<limit-1e-9:
                    return False
            return True

        def allowed(target):
            target = np.asarray(target, dtype=float)[:2]
            return ((max_anchor_distance is None
                     or np.linalg.norm(target - anchor) <= max_anchor_distance + 1e-9)
                    and np.linalg.norm(target - position) > .1
                    and not any(np.linalg.norm(target - old) < 1e-6 for old, _ in entries)
                    and self.segment_safe(position, target)
                    and peers_clear(target)
                    and (segment_filter is None or segment_filter(position,target)))

        def peer_side_prefix():
            directions=[]
            if preferred is not None:directions.append(np.asarray(preferred,dtype=float)[:2])
            graph_coords=getattr(agent,'node_coords',None)
            graph_indices=getattr(agent,'neighbor_indices',None)
            if graph_coords is not None and graph_indices is not None:
                graph_coords=np.asarray(graph_coords,dtype=float)
                directions.extend(graph_coords[int(i),:2] for i in np.unique(graph_indices) if 0<=int(i)<len(graph_coords))
            if base is not None:directions.append(np.asarray(base,dtype=float)[:2])
            blocked=False
            for target in directions:
                delta=target-position;distance=float(np.linalg.norm(delta))
                if distance<=.1:continue
                first=position+delta/distance*min(1.,distance)
                if ((self.segment_safe(position,target) and not peers_clear(target))
                        or (self.segment_safe(position,first) and not peers_clear(first))):
                    blocked=True;break
            if not blocked:return None
            candidates=[]
            for order,angle in enumerate(np.linspace(0.,2*np.pi,36,endpoint=False)):
                direction=np.array([np.cos(angle),np.sin(angle)])
                for length in (1.,.5,.25,.125):
                    target=position+length*direction
                    closing=float(np.linalg.norm(position-progress)-np.linalg.norm(target-progress))
                    if (closing>.05 or (prefer_base and abs(closing)<=.15)) and allowed(target):
                        candidates.append((-closing,-length,order,target));break
            return None if not candidates else min(candidates,key=lambda row:row[:3])[3].copy()

        if prefer_base:
            # No complete goal path: reuse the live exploration policy before
            # goal-greedy neighbors, through the same physical prefix gate.
            cached=self.prefixes.get(uid)
            if cached is not None and uid in self.prefix_headings and allowed(cached):
                return cached.copy()
            self.prefixes.pop(uid,None);self.prefix_headings.pop(uid,None)
            self.sensing_uids.discard(uid)
            if base is not None and allowed(base):
                target=np.asarray(base,dtype=float)[:2].copy()
                self.prefixes[uid]=target
                if base_heading is not None:self.prefix_headings[uid]=float(base_heading)
                return target.copy()
            if base is not None:
                delta=np.asarray(base,dtype=float)[:2]-position
                distance=float(np.linalg.norm(delta))
                if distance>.1:
                    for length in (1.,.5,.25,.125):
                        if length>=distance:continue
                        target=position+length*delta/distance
                        if allowed(target):
                            self.prefixes[uid]=target.copy()
                            self.prefix_headings[uid]=float(np.degrees(np.arctan2(delta[1],delta[0]))%360.)
                            self.sensing_uids.add(uid)
                            return target.copy()

        prefix = self.prefixes.get(uid)
        if prefix is not None:
            if allowed(prefix):
                return prefix.copy()
            self.prefixes.pop(uid, None)
            self.prefix_headings.pop(uid, None)

        self.sensing_uids.discard(uid)
        if preferred is not None and allowed(preferred):
            target = np.asarray(preferred, dtype=float)[:2].copy()
            if np.linalg.norm(position-progress)-np.linalg.norm(target-progress)<=.05:
                side=peer_side_prefix()
                if side is not None:
                    self.sensing_uids.add(uid);self.prefixes[uid]=side
                    return side.copy()
            self.prefixes[uid] = target
            return target.copy()

        # Reuse graph neighbors, while testing their prefix from the real pose.
        coords = getattr(agent, "node_coords", None)
        indices = getattr(agent, "neighbor_indices", None)
        if coords is not None:
            coords = np.asarray(coords, dtype=float)
            if indices is None:
                adjacency = getattr(agent, "adjacent_matrix", None)
                start = getattr(agent, "current_index", None)
                if adjacency is not None and start is not None:
                    indices = np.flatnonzero(np.asarray(adjacency)[int(start)] == 0)
            candidates = []
            for raw_index in np.unique(indices if indices is not None else []):
                index = int(raw_index)
                if not 0 <= index < len(coords):
                    continue
                target = coords[index, :2]
                length = float(np.linalg.norm(target - position))
                closing = float(np.linalg.norm(position - progress) - np.linalg.norm(target - progress))
                if .1 < length <= 6. and closing > .1 and allowed(target):
                    candidates.append((-closing, length, index, target))
            if candidates:
                target = min(candidates, key=lambda item: item[:3])[3].copy()
                self.prefixes[uid] = target
                return target.copy()

        side=peer_side_prefix()
        if side is not None:
            self.sensing_uids.add(uid);self.prefixes[uid]=side
            return side.copy()

        # Exploration fallback must pass exactly the same physical prefix gate.
        if base is not None and allowed(base):
            target = np.asarray(base, dtype=float)[:2].copy()
            self.prefixes[uid] = target
            if base_heading is not None:
                self.prefix_headings[uid] = float(base_heading)
            return target.copy()

        # A graph node can be beyond the current FoV although its first
        # supported movement is known free. Clip only existing waypoint
        # directions; this is a local prefix, not a new global planner.
        directions = []
        if preferred is not None:
            directions.append(np.asarray(preferred, dtype=float)[:2])
        if coords is not None and indices is not None:
            directions.extend(coords[int(i), :2] for i in np.unique(indices)
                              if 0 <= int(i) < len(coords))
        if base is not None:
            directions.append(np.asarray(base, dtype=float)[:2])
        if fallback_preferred is not None:
            directions.insert(0,np.asarray(fallback_preferred,dtype=float)[:2])
        partials = []
        for order, target in enumerate(directions):
            distance = float(np.linalg.norm(target - position))
            if distance <= .1:
                continue
            direction = (target - position) / distance
            for length in (1., .5, .25, .125):
                if length >= distance:
                    continue
                prefix = position + length * direction
                closing = float(np.linalg.norm(position - progress) - np.linalg.norm(prefix - progress))
                if allowed(prefix):
                    partials.append((-closing, -length, order, prefix))
                    break
        if partials:
            target = min(partials, key=lambda item: item[:3])[3].copy()
            self.sensing_uids.add(uid)
            self.prefixes[uid] = target
            return target.copy()

        return position.copy()
