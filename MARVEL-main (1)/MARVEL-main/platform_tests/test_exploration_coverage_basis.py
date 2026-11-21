import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.task_manager import TaskManager

def manager(basis=None):
    params={'target_coverage':.8}
    if basis is not None:params['coverage_basis']=basis
    return TaskManager([{'task_id':'T1','type':'exploration','priority':3,'params':params}])

def test_explicit_observed_free_basis_can_complete_true_coverage():
    tasks=manager('observed_free')
    tasks.update(0,[],{},exploration_rate=.49,observed_free_rate=.9)
    assert tasks.tasks['T1'].status=='complete'
    assert tasks.tasks['T1'].progress==.9

def test_legacy_visible_basis_keeps_existing_completion_semantics():
    tasks=manager()
    tasks.update(0,[],{},exploration_rate=.49,observed_free_rate=.9)
    assert tasks.tasks['T1'].status=='active'
    assert tasks.tasks['T1'].progress==.49

def test_search_activation_still_uses_legacy_protocol_quantity():
    tasks=TaskManager([{'task_id':'T2','type':'target_search','priority':5,'activation':{'condition':'exploration_rate >= .3'},'params':{'target_count':1}}])
    tasks.update(0,[],{},exploration_rate=.29,observed_free_rate=.9,target_detections={})
    assert tasks.tasks['T2'].status=='pending'
