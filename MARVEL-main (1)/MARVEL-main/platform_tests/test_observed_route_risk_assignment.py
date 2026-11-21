from platform_tests.test_observed_matching_assignment import graph
from scripts.observed_route_risk_assignment import route_risk_assignment

def estimates():return {(100,0):{'unknown_residual_lower_bound_m':8.},(100,1):{'unknown_residual_lower_bound_m':0.}}
def setup():
    g=graph([[10,12]])
    for u in g.uav_states:u.velocity=1.
    return g

def test_existing_observed_unknown_residual_changes_risk_rank():
    a=route_risk_assignment(setup(),estimates());assert a.uav_id==1

def test_no_extra_diagnostics_falls_back_to_exact_rule_cost():
    a=route_risk_assignment(setup(),{});assert a.uav_id==0

def test_uncertainty_never_overrides_legal_mask():
    g=setup();g.action_mask[0,1]=True
    assert route_risk_assignment(g,estimates()).uav_id==0

def test_zero_penalty_reproduces_cost_ranking():
    assert route_risk_assignment(setup(),estimates(),0).uav_id==0
