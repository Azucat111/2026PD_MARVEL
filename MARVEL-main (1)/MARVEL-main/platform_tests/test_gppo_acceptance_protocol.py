import pytest
from scripts.gppo_acceptance_protocol import evaluation_stage

def test_development_is_explicitly_not_final():
 assert evaluation_stage(9155,30,80,False,'static')=='development rollout, not final 15% acceptance'

def test_reserved_seeds_cannot_be_used_for_development():
 with pytest.raises(ValueError):evaluation_stage(101101,30,300,False,'static')

@pytest.mark.parametrize('seed,uavs,steps',[(9155,30,300),(101101,4,300),(101101,60,80)])
def test_final_rejects_unreserved_or_inadequate_scope(seed,uavs,steps):
 with pytest.raises(ValueError):evaluation_stage(seed,uavs,steps,True,'dynamic')

@pytest.mark.parametrize('domain',['static','dynamic'])
def test_heldout_stage_requires_explicit_valid_final_invocation(domain):
 assert evaluation_stage(202202,60,300,True,domain)=='heldout final Search acceptance'
