from types import SimpleNamespace
import numpy as np
import pytest
from utils.geometry import GeometryFrame
from scripts.observed_search_visibility import ObservedSearchVisibility,visibility_assignment
from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType

def observation(cells,mode='extended'):
    return {'visible_occupancy':cells,'occupancy_frame':GeometryFrame(0,0,1,12,12,mode).describe(),'visible_occupancy_semantics':'areal_file_grid'}

def test_fraction_uses_current_own_sensor_cells_and_public_support_only():
    tracker=ObservedSearchVisibility(GeometryFrame(0,0,1,12,12),1.)
    tracker.observe({1:observation([[5,5,0],[5,4,0],[5,6,0]]),2:observation([[6,5,0]])})
    assert tracker.features([5,5],1)==pytest.approx((.6,1.))
    assert tracker.features([5,5],2)==pytest.approx((.2,1.))

def test_only_observed_static_occupied_cells_are_removed():
    tracker=ObservedSearchVisibility(GeometryFrame(0,0,1,12,12),1.)
    tracker.observe({1:observation([[5,5,0],[5,4,0],[5,6,0]]),2:observation([[4,5,1]])})
    assert tracker.features([5,5],1)==pytest.approx((.75,1.))

def test_native_missing_or_mismatched_metadata_fail_without_visibility_claim():
    tracker=ObservedSearchVisibility(GeometryFrame(0,0,1,12,12),1.)
    tracker.observe({1:{},2:observation([[5,5,0]],mode='marvel_native')})
    assert tracker.features([5,5],1)==(0.,0.)
    assert tracker.features([5,5],2)==(0.,0.)

def test_visibility_probe_respects_priority_and_masks_and_falls_back_to_rule():
    tracker=ObservedSearchVisibility(GeometryFrame(0,0,1,12,12),1.)
    tracker.observe({0:observation([[5,5,0]]),1:observation([[5,5,0],[5,4,0],[5,6,0]])})
    graph=TaskGraph([UAVState(0,position=(5,5)),UAVState(1,position=(8,5))],[Subtask(10,TaskType.TARGET_SEARCH,5,position=(5,5))])
    assert visibility_assignment(graph,tracker).uav_id==1
    graph.action_mask[0,1]=True
    assert visibility_assignment(graph,tracker).uav_id==0
    tracker.observe({})
    graph.action_mask[0,1]=False
    assert visibility_assignment(graph,tracker).uav_id==0

class NoPrivateReads(dict):
    def get(self,key,*args):
        if key in ('hidden_targets','survivors','ground_truth','true_map'):raise AssertionError('private read')
        return super().get(key,*args)

def test_private_truth_keys_are_never_consumed():
    tracker=ObservedSearchVisibility(GeometryFrame(0,0,1,12,12),1.)
    tracker.observe({1:NoPrivateReads(observation([[5,5,0]]))})
    assert tracker.features([5,5],1)==pytest.approx((.2,1.))

def test_visibility_uses_shared_public_state_age_fraction():
 tracker=ObservedSearchVisibility(GeometryFrame(0,0,1,12,12),1.,age_fraction=.05)
 tracker.observe({1:observation([[5,5,0],[5,4,0],[5,6,0]])})
 tracker.observe({1:observation([[6,5,0]])})
 assert tracker.features([5,5],1)==pytest.approx((.22,1.))
