import math
import pytest
from scripts.gppo_benchmark_domains import load_domain,scenario_paths

def fresh_loader(file):
    return {'scenario':{'duration':{'step_dt':.1}}}

@pytest.mark.parametrize('speed,expected',[(.2,.2),(1.5,1.5)])
def test_declared_physical_speed_converts_to_mission_units(speed,expected):
    cfg=load_domain(fresh_loader,'ignored.yaml',dynamic=True,warning_rate_mps=speed)
    assert cfg['dynamic_obstacles'][0]['params']['expansion_rate']==pytest.approx(expected)

@pytest.mark.parametrize('speed',[0,-1,math.inf,math.nan])
def test_invalid_speed_rejected(speed):
    with pytest.raises(ValueError):load_domain(fresh_loader,'ignored.yaml',dynamic=True,warning_rate_mps=speed)

def test_static_does_not_add_hazard_and_arms_are_independent():
    first=load_domain(fresh_loader,'ignored.yaml',dynamic=True,warning_rate_mps=.2)
    second=load_domain(fresh_loader,'ignored.yaml')
    assert 'dynamic_obstacles' not in second
    first['scenario']['duration']['step_dt']=7
    assert second['scenario']['duration']['step_dt']==.1

def test_override_paths_exist_and_preserve_separate_original_map():
    paths=scenario_paths('configs/scenarios/extended_connected_gppo_20261010.yaml')
    assert len(paths)==2 and all(p.exists() for p in paths)
    assert paths[1].name=='extended_connected_gppo_20261010.png'


@pytest.mark.parametrize('speed',[.2,1.5])
@pytest.mark.parametrize('dt',[.05,.1,.2])
def test_full_gppo_pipeline_preserves_declared_metres_per_second(speed,dt):
    from integrations.gppo.config_profile import prepare_gppo_config
    from utils.obstacle_manager import ExpandingCircleObstacle
    import numpy as np
    cp={'mixed_config':dict(control_step_ms=1000,max_episode_steps=300,comm_range=20,comm_max_hops=2,comm_delay_ms=50,activation_threshold=.3,connectivity_threshold=.9)}
    def loader(file):return {'scenario':{'duration':{'step_dt':dt}},'tasks':[]}
    cfg=prepare_gppo_config(load_domain(loader,'unused.yaml',dynamic=True,warning_rate_mps=speed),cp)
    from scripts.gppo_benchmark_domains import validate_physical_warning_rate
    validate_physical_warning_rate(cfg)
    raw=cfg['dynamic_obstacles'][0];params=raw['params']
    obstacle=ExpandingCircleObstacle(raw['id'],np.array(params['position']),params['initial_radius'],params['expansion_rate'],params['max_radius'],raw['spawn_step'])
    obstacle.step(raw['spawn_step']+round(2/dt),dt)
    assert obstacle.radius==pytest.approx(min(4.,speed*2.))


def test_wrong_frozen_control_period_rejected_before_running():
    from scripts.gppo_benchmark_domains import validate_physical_warning_rate
    cfg={'_benchmark_warning_units':dict(control_step_seconds=1.,physical_rate_mps=.2),'_gppo_profile':dict(control_step_ms=2000),'scenario':{'duration':{'step_dt':.1}}}
    with pytest.raises(ValueError,match='control period'):validate_physical_warning_rate(cfg)

def test_resolved_double_scaled_rate_rejected_before_running():
    from scripts.gppo_benchmark_domains import validate_physical_warning_rate
    cfg={'_benchmark_warning_units':dict(control_step_seconds=1.,physical_rate_mps=.2),'_gppo_profile':dict(control_step_ms=1000),'scenario':{'duration':{'step_dt':.1}},'dynamic_obstacles':[dict(id='natural_expanding_warning',params=dict(expansion_rate=.002))]}
    with pytest.raises(ValueError,match='expansion rate'):validate_physical_warning_rate(cfg)

def test_legacy_config_is_not_reinterpreted():
    from scripts.gppo_benchmark_domains import validate_physical_warning_rate
    validate_physical_warning_rate({'dynamic_obstacles':[]})
