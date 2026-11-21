import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('budget_wrapper',Path(__file__).resolve().parents[1]/'scripts/train_extended_gppo_wall_budget.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_only_wall_budget_changes():
    seen={}
    def original(*args,**kwargs):seen.update(args=args,kwargs=kwargs);return 'same_result'
    wrapped=m.bind_wall_budget(original,1800.)
    assert wrapped('folder','gppo',60,61244,80,physical_coverage=True,stop_on_completion=False)=='same_result'
    assert seen==dict(args=('folder','gppo',60,61244,80),kwargs=dict(physical_coverage=True,stop_on_completion=False,wall_budget_seconds=1800.))
@pytest.mark.parametrize('budget',[0.,-1.,float('inf'),float('nan')])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):m.bind_wall_budget(lambda:None,budget)
def test_conflicting_override_rejected():
    with pytest.raises(ValueError):m.bind_wall_budget(lambda **kwargs:None,1800.)(wall_budget_seconds=600.)
