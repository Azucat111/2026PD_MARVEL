"""True Search detection metric; heat service is not target detection."""
import math

def detection_efficiency(events,target_count,dt,horizon,task_id='T2_target_search'):
    if target_count<=0 or dt<=0 or horizon<=0:
        raise ValueError('positive target_count, dt and horizon required')
    times=[float(horizon)]*int(target_count)
    found=set()
    for event in events:
        data=event.get('data',{})
        if (event.get('type')!='target_found' or data.get('task_id')!=task_id
                or data.get('detection_mode')!='sensor_visible_cell'):
            continue
        index=data.get('target_index')
        elapsed=(int(event['step'])+1)*float(dt)
        if isinstance(index,int) and 0<=index<target_count and 0<=elapsed<=horizon:
            times[index]=min(times[index],elapsed)
            found.add(index)
    return {'capped_detection_seconds':times,'detected_targets':len(found),
        'total_targets':target_count,'horizon_seconds':float(horizon),
        'mean_capped_seconds':math.fsum(times)/target_count}


def observed_free_coverage(explored_cells,occupancy_grid):
    """Evaluation oracle only: observed FREE / all physical FREE cells.

    Do not hand this truth-grid input to routing or assignment decisions.
    It preserves the separate legacy activation/coverage control quantity.
    """
    import numpy as np
    grid=np.asarray(occupancy_grid)
    if grid.ndim!=2:
        raise ValueError('2D occupancy grid required')
    total=int(np.count_nonzero(grid==0))
    observed=sum(1 for col,row in set(explored_cells)
        if 0<=int(col)<grid.shape[1] and 0<=int(row)<grid.shape[0] and grid[int(row),int(col)]==0)
    return {'observed_free_cells':observed,'total_free_cells':total,
        'coverage':observed/total if total else 1.,'evaluation_oracle_only':True}
