"""Stop at first dwell completion: sensor-oracle diagnosis, never policy input."""
import json,numpy as np
import benchmark_fair_search as bench
OriginalRuntime=bench.SimulationRuntime
out=bench.ROOT/'artifacts/search_unconfirmed_service_20261010_attempt3';out.mkdir(exist_ok=False)
class CapturedService(BaseException):pass
class TracingRuntime(OriginalRuntime):
    def step(self,actions):
        robot=next(r for r in self.robots if r.robot_id==15)
        before=robot.position.copy();before_heading=float(robot.heading)
        observations,info=super().step(actions)
        truth=self.search_scenario
        survivor=next(t for t in truth.survivors if t.heat_id==0)
        target=np.asarray(survivor.position)
        cell=tuple(int(round(value)) for value in self.obstacles.frame.world_to_cell_float(target))
        sensor=self.sensor_by_type.get(robot.robot_type,self.sensor)
        delta=target-robot.position
        angle=float((np.degrees(np.arctan2(delta[1],delta[0]))-robot.heading+180)%360-180)
        row={'step':self.current_step,'position':robot.position.tolist(),'heading':float(robot.heading),
            'before_heading':before_heading,'moved_m':float(np.linalg.norm(robot.position-before)),
            'requested_action':[np.asarray(actions[15][0]).tolist(),float(actions[15][1])],
            'feedback':self._motion_feedback.get(15),'oracle_only_target':target.tolist(),
            'oracle_heat':next(h.position for h in truth.heat_points if h.heat_id==0),
            'target_index':survivor.target_index,'distance_m':float(np.linalg.norm(delta)),
            'angle_from_heading_deg':angle,'within_fov':abs(angle)<=sensor.fov/2,
            'los_clear_oracle':sensor._has_line_of_sight(robot.position,cell,self.obstacles.get_occupancy_grid()),
            'target_cell_visible':any(np.array_equal(c,cell) for c in observations[15]['visible_cells'])}
        with (out/'ticks.jsonl').open('a') as handle:handle.write(json.dumps(row,default=lambda x:x.tolist())+'\n')
        completions=[e for e in self.events if e['type']=='gppo_search_heat_serviced' and e['data'].get('heat_id')==0]
        if completions:
            (out/'captured_service.json').write_text(json.dumps({'event':completions[-1],
                'final_row':row,'completed_heat_feedback':sorted(self.completed_search_heat_ids()),
                'truth_logged_only_for_diagnosis':True,'production_modified':True},indent=2,default=lambda x:x.tolist()))
            raise CapturedService()
        return observations,info
bench.SimulationRuntime=TracingRuntime
try:bench.run(out,'rule',60,78901,8,0,True,False)
except CapturedService:print((out/'captured_service.json').read_text(),flush=True)
finally:bench.SimulationRuntime=OriginalRuntime
