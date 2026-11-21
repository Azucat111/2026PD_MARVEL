import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.search_efficiency import detection_efficiency

def event(index,step,mode='sensor_visible_cell'):
    return {'step':step,'type':'target_found','data':{'target_index':index,'task_id':'T2_target_search','detection_mode':mode}}

def test_counts_true_first_sensor_detection_and_censors_missing_targets():
    result=detection_efficiency([event(0,19),event(0,29),event(1,9,'legacy_radius')],3,.1,30)
    assert result['capped_detection_seconds']==[2.,30.,30.]
    assert result['detected_targets']==1
    assert result['mean_capped_seconds']==62/3

def test_excludes_future_and_other_task_events():
    e=event(1,9);e['data']['task_id']='other'
    result=detection_efficiency([event(0,399),e],2,.1,30)
    assert result['detected_targets']==0

def test_free_coverage_is_evaluation_only_and_excludes_seen_obstacles():
    import numpy as np
    from integrations.search_efficiency import observed_free_coverage
    result=observed_free_coverage({(0,0),(1,0),(-1,0),(5,5)},np.array([[0,1],[0,0]],dtype=np.uint8))
    assert result['observed_free_cells']==1
    assert result['total_free_cells']==3
    assert result['coverage']==1/3
