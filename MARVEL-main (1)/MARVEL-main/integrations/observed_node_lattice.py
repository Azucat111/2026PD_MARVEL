"""Extended MARVEL node lattice selection with explicit empty FREE components.

Same lattice, ordering and connected-component primitive as frozen MARVEL.
No truth-map access, node invention or change to the native utility helper.
"""
import numpy as np
from parameter import CELL_SIZE,NODE_RESOLUTION,FREE
from utils.utils import get_cell_position_from_coords,get_free_and_connected_map

def extended_updating_node_coords(location,updating_map_info,check_connectivity=True):
    info=updating_map_info
    x_min,y_min=info.map_origin_x,info.map_origin_y
    x_max=x_min+(info.map.shape[1]-1)*CELL_SIZE
    y_max=y_min+(info.map.shape[0]-1)*CELL_SIZE
    if x_min%NODE_RESOLUTION!=0:x_min=(x_min//NODE_RESOLUTION+1)*NODE_RESOLUTION
    if x_max%NODE_RESOLUTION!=0:x_max=x_max//NODE_RESOLUTION*NODE_RESOLUTION
    if y_min%NODE_RESOLUTION!=0:y_min=(y_min//NODE_RESOLUTION+1)*NODE_RESOLUTION
    if y_max%NODE_RESOLUTION!=0:y_max=y_max//NODE_RESOLUTION*NODE_RESOLUTION
    xx,yy=np.meshgrid(np.arange(x_min,x_max+.1,NODE_RESOLUTION),
                     np.arange(y_min,y_max+.1,NODE_RESOLUTION))
    nodes=np.around(np.vstack([xx.T.ravel(),yy.T.ravel()]).T,1)
    connected=None
    if check_connectivity:
        source=get_cell_position_from_coords(np.asarray(location),info)
        # Label zero is UNKNOWN/OCCUPIED background, never a free component.
        connected=(get_free_and_connected_map(location,info) if info.map[source[1],source[0]]==FREE
                   else np.zeros(info.map.shape,dtype=bool))
        allowed=connected
    else:allowed=info.map==FREE
    cells=get_cell_position_from_coords(nodes,info).reshape(-1,2)
    assert np.all((cells[:,0]>=0)&(cells[:,0]<info.map.shape[1]))
    assert np.all((cells[:,1]>=0)&(cells[:,1]<info.map.shape[0]))
    # flatnonzero is an integer index even when no grid node lies in a
    # measured FREE component. Existing shared nodes remain untouched.
    indices=np.flatnonzero(allowed[cells[:,1],cells[:,0]])
    return nodes[indices].reshape(-1,2),connected
