"""Extended empty connected node selection must not change frozen native helper."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parameter import FREE,UNKNOWN
from utils.utils import MapInfo,get_updating_node_coords
from utils.node_manager import NodeManager

def captured_map():
    evidence=json.loads((ROOT/'artifacts/empty_observed_graph_20261010_attempt2/failures.json').read_text())[0]
    return np.asarray(evidence['position']),MapInfo(np.asarray(evidence['map']),*evidence['origin'],evidence['cell_size'])

def test_real_empty_connected_lattice_extended_update_is_valid_noop():
    position,info=captured_map()
    manager=NodeManager(120,10);manager.use_clearance_admission=True
    manager.clearance_blocked=lambda point:False
    manager.update_graph(position,np.empty((0,2)),info,info)
    assert len(manager.nodes_dict)==0

def test_native_helper_literal_boundary_still_matches_frozen_failure():
    import pytest
    position,info=captured_map()
    with pytest.raises(IndexError,match='indices'):
        get_updating_node_coords(position,info)

def test_extended_nonempty_lattice_matches_existing_order_and_component():
    from integrations.observed_node_lattice import extended_updating_node_coords
    for origin in ((0.,0.),(-8.4,-12.4),(3.6,7.6)):
        grid=np.full((41,41),FREE,dtype=int)
        grid[10:20,12:14]=UNKNOWN
        info=MapInfo(grid,*origin,.4)
        position=np.asarray(origin)+[2.,2.]
        for connectivity in (True,False):
            expected=get_updating_node_coords(position,info,connectivity)
            actual=extended_updating_node_coords(position,info,connectivity)
            np.testing.assert_array_equal(actual[0],expected[0])
            if connectivity:np.testing.assert_array_equal(actual[1],expected[1])

def test_extended_background_is_not_free_connected_component():
    from integrations.observed_node_lattice import extended_updating_node_coords
    grid=np.full((31,31),UNKNOWN,dtype=int);grid[20:,20:]=FREE
    info=MapInfo(grid,0.,0.,.4)
    nodes,component=extended_updating_node_coords(np.array([2.,2.]),info)
    assert nodes.shape==(0,2)
    assert not component.any()
