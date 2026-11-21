import pytest
from scripts.score_gppo_pairs import aggregate_cases

def case(seed,uavs,domain,gppo,rule,sha='one',horizon=80):
 return dict(seed=seed,uavs=uavs,domain=domain,checkpoint_sha256=sha,source_revision='same-production',horizon_seconds=horizon,final_eligible=True,valid=True,collisions=0,targets=4,found=4,gppo_total=4*gppo,rule_total=4*rule)

def test_efficiency_uses_pooled_latency_not_average_case_percentages():
 result=aggregate_cases([case(1,30,'static',1,2),case(2,60,'static',10,9)])
 assert result['efficiency_gain']==pytest.approx(0.)
 assert not result['final_search_metric_acceptance']

def test_mixed_model_pooling_and_duplicate_cases_are_rejected():
 with pytest.raises(ValueError):aggregate_cases([case(1,30,'static',1,2),case(2,60,'static',1,2,'other')])
 with pytest.raises(ValueError):aggregate_cases([case(1,30,'static',1,2),case(1,30,'static',1,2)])

def test_high_gain_smoke_never_satisfies_final_contract():
 rows=[case(s,n,d,1,2) for s in [101101,202202,303303] for n in [30,60] for d in ['static','dynamic']]
 assert not aggregate_cases(rows)['final_search_metric_acceptance']
 for r in rows:r['horizon_seconds']=300
 assert aggregate_cases(rows)['final_search_metric_acceptance']
 rows[0]['collisions']=1
 assert not aggregate_cases(rows)['final_search_metric_acceptance']

def test_same_model_cannot_pool_different_production_revisions():
 rows=[case(1,30,'static',1,2),case(2,60,'dynamic',1,2)]
 rows[1]['source_revision']='changed-motion'
 with pytest.raises(ValueError):aggregate_cases(rows)

def test_final_claim_requires_source_provenance():
 rows=[case(s,n,d,1,2,horizon=300) for s in [101101,202202,303303] for n in [30,60] for d in ['static','dynamic']]
 for row in rows:row.pop('source_revision')
 assert not aggregate_cases(rows)['final_search_metric_acceptance']

def test_experimental_controller_is_not_pooled_with_production():
 from scripts.score_gppo_pairs import shared_source_fingerprint
 base={'source_hashes':{'integrations/a.py':'same','scripts/benchmark_fair_search.py':'same'}}
 treatment={**base,'controller_variant':'postcompletion_hold','source_hashes':{**base['source_hashes'],'scripts/completed_task_stationkeeping.py':'hold-v1'}}
 assert shared_source_fingerprint(base)!=shared_source_fingerprint(treatment)
 changed={**treatment,'source_hashes':{**treatment['source_hashes'],'scripts/completed_task_stationkeeping.py':'hold-v2'}}
 assert shared_source_fingerprint(treatment)!=shared_source_fingerprint(changed)

def test_actor_input_and_benchmark_loader_changes_affect_provenance():
 from scripts.score_gppo_pairs import shared_source_fingerprint
 base={'source_hashes':{'integrations/a.py':'same','scripts/extended_gppo_heading.py':'v2','scripts/gppo_benchmark_domains.py':'mps-v1'}}
 for key in ('scripts/extended_gppo_heading.py','scripts/gppo_benchmark_domains.py'):
  changed={'source_hashes':{**base['source_hashes'],key:'changed'}}
  assert shared_source_fingerprint(base)!=shared_source_fingerprint(changed)


def physical_config(rate):
 return {'scenario':{'duration':{'step_dt':.1}},'dynamic_obstacles':[{'id':'natural_expanding_warning','params':{'expansion_rate':rate}}]}

def test_pair_score_rejects_mislabeled_dynamic_speed():
 from scripts.score_gppo_pairs import validate_pair_warning_units
 with pytest.raises(ValueError,match='physical warning'):validate_pair_warning_units({'warning_rate_mps':.2},physical_config(.002))

def test_pair_score_accepts_verified_dynamic_speed():
 from scripts.score_gppo_pairs import validate_pair_warning_units
 assert validate_pair_warning_units({'warning_rate_mps':.2},physical_config(.02))==pytest.approx(.2)

def test_final_dynamic_without_declared_speed_is_rejected():
 from scripts.score_gppo_pairs import validate_pair_warning_units
 with pytest.raises(ValueError,match='declaration'):validate_pair_warning_units({'stage':'heldout final Search acceptance'},physical_config(.02))

def test_legacy_static_pair_needs_no_dynamic_conversion():
 from scripts.score_gppo_pairs import validate_pair_warning_units
 assert validate_pair_warning_units({}, {}) is None
