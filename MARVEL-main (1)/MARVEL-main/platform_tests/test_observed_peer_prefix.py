from types import SimpleNamespace as NS
import numpy as np
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.geometry import GeometryFrame

def test_occupied_graph_nodes_do_not_force_backtracking_when_legal_side_prefix_exists():
    rec=ObservedSearchRecovery(GeometryFrame.extended(width_cells=80,height_cells=80))
    rec.cells={(c,r):0 for c in range(80) for r in range(80)}
    position=np.array([15.6,11.7]);anchor=np.array([36.5,16.5])
    rec.nearby_positions[4]=[np.array(p) for p in [[16.,12.],[16.,8.],[12.,12.],[12.,8.]]]
    agent=NS(node_coords=np.array([[20.,12.],[20.,8.],[12.,8.],[12.,12.]]),neighbor_indices=np.arange(4))
    target=rec.select(4,position,anchor,agent,np.array([20.,12.]),np.array([12.,8.]))
    assert rec.segment_safe(position,target)
    assert np.linalg.norm(target-anchor)<np.linalg.norm(position-anchor)-.05
    delta=target-position
    for peer in rec.nearby_positions[4]:
        fraction=np.clip(np.dot(peer-position,delta)/np.dot(delta,delta),0.,1.)
        assert np.linalg.norm(position+fraction*delta-peer)>=.5-1e-9


def test_legal_backtracking_graph_hop_does_not_preempt_certified_peer_bypass():
    rec=ObservedSearchRecovery(GeometryFrame.extended(width_cells=80,height_cells=80))
    rec.cells={(c,r):0 for c in range(80) for r in range(80)}
    position=np.array([12.456740080647897,10.050419067521759]);anchor=np.array([38.5,16.5])
    rec.nearby_positions[4]=[np.array(p) for p in [[12.,8.],[16.,8.],[20.,8.],[8.,12.],[12.,12.],[16.,12.],[20.,12.],[8.,16.],[12.,16.],[16.,16.],[20.,16.],[12.,20.]]]
    agent=NS(node_coords=np.array([[8.,8.],[12.,8.],[12.,12.],[16.,8.],[16.,12.],[20.,8.],[20.,12.]]),neighbor_indices=np.arange(7))
    selected=rec.select(4,position,anchor,agent,np.array([8.,8.]),np.array([8.,8.]))
    assert rec.segment_safe(position,selected)
    assert np.linalg.norm(selected-anchor)<np.linalg.norm(position-anchor)-.05
