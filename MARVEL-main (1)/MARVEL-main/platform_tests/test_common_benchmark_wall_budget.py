"""Requested physical windows must not inherit an unchangeable wall cutoff."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_fair_search as bench

@pytest.fixture
def tiny_benchmark(monkeypatch):
    clock={'now':0.}
    def monotonic():
        value=clock['now'];clock['now']+=200.;return value
    monkeypatch.setattr(bench.time,'monotonic',monotonic)
    monkeypatch.setattr(bench,'load_and_validate_scenario',lambda _: {'scenario':{},'tasks':[]})
    monkeypatch.setattr(bench,'load_frozen_gppo',lambda *a,**k:(None,{}))
    monkeypatch.setattr(bench,'prepare_gppo_config',lambda config,_:config)
    monkeypatch.setattr(bench,'validate_physical_warning_rate',lambda _:None)
    monkeypatch.setattr(bench,'observed_free_coverage',lambda *a:{})
    class Runtime:
        def __init__(self,config):
            self.current_step=0;self.dt=1.;self.robots=[];self.events=[];self.exploration_rate=0.
            self.tasks=NS(all_complete=lambda:False,summary=lambda:{})
            self.search_scenario=NS(target_count=1);self.explored_cells=np.zeros((1,1))
            self.obstacles=NS(get_occupancy_grid=lambda:np.zeros((1,1)))
        def reset(self):return {}
        def _mission_step_interval(self):return 1
        def step(self,actions):self.current_step+=1;return {},{'collisions':[]}
    class Adapter:
        def __init__(self,runtime):
            self._using_policy=True
            self.scheduler=NS(allocator=NS(adapter=NS(assign=lambda *a,**k:None)),heat_points=[],search_activated=True,
                serviced_heat_ids=set(),search_assignments={},relay_assignments={},search_activation_step=0)
        def setup(self):pass
        def get_actions(self,obs):return {}
    monkeypatch.setattr(bench,'SimulationRuntime',Runtime)
    monkeypatch.setattr(bench,'MARVELPolicyAdapter',Adapter)
    return clock

@pytest.mark.parametrize('budget,expected',[(600.,4),(2000.,6)])
def test_explicit_budget_controls_actual_run_window(tmp_path,tiny_benchmark,budget,expected):
    result=bench.run(tmp_path,'rule',30,9163,6,wall_budget_seconds=budget)
    assert result['mission_steps']==expected
    assert result['elapsed_simulation_seconds']==expected
    assert result['valid_for_comparison']==(expected==6)
    assert result['wall_budget_seconds']==budget

@pytest.mark.parametrize('budget',[0.,-1.,float('inf'),float('nan')])
def test_invalid_budget_rejected_before_runtime(tmp_path,monkeypatch,budget):
    monkeypatch.setattr(bench,'load_and_validate_scenario',lambda _:pytest.fail('Invalid budget reached runtime setup'))
    with pytest.raises(ValueError):bench.run(tmp_path,'rule',30,9163,6,wall_budget_seconds=budget)
    assert list(tmp_path.iterdir())==[]
