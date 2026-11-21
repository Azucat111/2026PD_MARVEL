"""Explicit paired benchmark domain inputs; no scheduler truth-map access."""
from pathlib import Path
import math
ROOT=Path(__file__).resolve().parents[1]

def load_domain(base_loader,file,scenario_override=None,dynamic=False,warning_rate_mps=1.5,control_step_seconds=1.):
    selected=file if scenario_override is None else Path(scenario_override)
    if not Path(selected).is_absolute():selected=ROOT/selected
    config=base_loader(selected)
    if dynamic:
        if not math.isfinite(warning_rate_mps) or warning_rate_mps<=0:raise ValueError('Positive finite physical warning expansion speed required')
        if not math.isfinite(control_step_seconds) or control_step_seconds<=0:raise ValueError('Positive finite mission control period required')
        # prepare_gppo_config converts mission-index rate to physics-index rate.
        # Supply metres per mission tick here; never pre-scale by physics dt.
        config['_benchmark_warning_units']={'physical_rate_mps':float(warning_rate_mps),'control_step_seconds':float(control_step_seconds)}
        config['dynamic_obstacles']=[dict(id='natural_expanding_warning',type='expanding_circle',spawn_step=1,
            params=dict(position=[20.,20.],initial_radius=0.,expansion_rate=warning_rate_mps*control_step_seconds,max_radius=4.))]
    return config

def scenario_paths(scenario_override):
    if scenario_override is None:return []
    from utils.scenario_config import load_and_validate_scenario
    selected=Path(scenario_override)
    if not selected.is_absolute():selected=ROOT/selected
    config=load_and_validate_scenario(selected)
    return [selected,Path(config['environment']['map_file'])]


def validate_physical_warning_rate(config):
    """Check resolved physics units before runtime/reset; legacy configs unchanged."""
    declared=config.get('_benchmark_warning_units')
    if declared is None:return
    dt=float(config['scenario']['duration']['step_dt'])
    period=float(config['_gppo_profile']['control_step_ms'])/1000.
    if not math.isclose(period,declared['control_step_seconds'],rel_tol=1e-9,abs_tol=1e-12):
        raise ValueError('Benchmark mission period does not match frozen control period')
    rows=[o for o in config.get('dynamic_obstacles',[]) if o['id']=='natural_expanding_warning']
    if len(rows)!=1:raise ValueError('Expected one declared benchmark warning')
    actual=float(rows[0]['params']['expansion_rate'])/dt
    if not math.isclose(actual,declared['physical_rate_mps'],rel_tol=1e-9,abs_tol=1e-12):
        raise ValueError('Resolved physical warning expansion rate disagrees with declaration')
