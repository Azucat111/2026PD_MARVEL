"""One bounded instrumented30-UAV trace for a newly observed recovery failure."""
from pathlib import Path
import argparse,json
import benchmark_fair_search as bench
from integrations.gppo.observed_graph import _graph_state
from integrations.gppo.observed_reachability import observed_search_impossibility
from integrations.gppo.search_deadline import certified_constant_search_capability
OriginalRuntime=bench.SimulationRuntime
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=40);args=p.parse_args()
out=bench.ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
class TracingRuntime(OriginalRuntime):
    def step(self,actions):
        obs,info=super().step(actions)
        scheduler=getattr(self,'high_level_scheduler',None)
        if scheduler is not None:
            entries=[]
            for heat,uid in scheduler.search_assignments.items():
                robot=next(r for r in self.robots if r.robot_id==uid)
                entries.append({'step':self.current_step,'uid':uid,'heat_id':heat,
                    'position':robot.position.tolist(),'heading':robot.heading,
                    'goal':scheduler.heat_by_id[heat].position,
                    'feedback':self._motion_feedback.get(uid)})
            with (out/'search_ticks.jsonl').open('a') as handle:
                for entry in entries:handle.write(json.dumps(entry,default=lambda x:x.tolist())+'\n')
            if self.current_step%100==0:
                controller=scheduler._search_recovery
                state={'step':self.current_step,'cells':[[col,row,value] for (col,row),value in controller.cells.items()],
                    'areal_grid':getattr(controller,'_authoritative_areal_grid',None),
                    'capabilities':{e['uid']:certified_constant_search_capability(self,e['uid']) for e in entries},
                    'direct_reasons':{e['uid']:observed_search_impossibility(self,e['uid'],e['position'],e['goal'],.05) for e in entries},
                    'entries':entries,'prefixes':controller.prefixes,'anchors':controller.anchors,'failed':controller.failed,'agents':{}}
                for entry in entries:
                    uid=entry['uid'];agent=scheduler._marvel_agents[uid];graph=_graph_state(agent)
                    state['agents'][uid]=None if graph is None else {'coords':graph[0],'adj':graph[1],'start':graph[2],
                        'neighbors':agent.neighbor_indices}
                (out/f'snapshot_{self.current_step}.json').write_text(json.dumps(state,default=lambda x:x.tolist()))
        return obs,info
bench.SimulationRuntime=TracingRuntime
result=bench.run(out,'gppo',30,45678,args.steps,0,True,False)
(out/'trace_summary.json').write_text(json.dumps(result,indent=2))
