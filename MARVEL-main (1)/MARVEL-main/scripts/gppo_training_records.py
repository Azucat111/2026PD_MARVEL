"""JSON-safe actual actor input snapshots, distinct from original graph features."""
import torch

def actor_input_record(batch,*,time,action,uav_ids,task_ids,log_prob=None,value=None):
    mask=batch['action_mask']
    if mask.shape!=(1,len(task_ids),len(uav_ids)):raise ValueError('Actor ID/mask mismatch')
    action=int(action);t,u=divmod(action,len(uav_ids))
    if not 0<=t<len(task_ids) or bool(mask[0,t,u]):raise ValueError('Invalid recorded action')
    snapshots={}
    for key,tensor in batch.items():
        if not bool(torch.isfinite(tensor).all()):raise ValueError('Nonfinite actor input')
        snapshots[key]=tensor.detach().cpu().tolist()
    return dict(time=float(time),action=action,uav_ids=list(uav_ids),task_ids=list(task_ids),
        chosen=dict(uav_id=int(uav_ids[u]),subtask_id=int(task_ids[t])),batch=snapshots,
        log_prob=None if log_prob is None else float(log_prob),value=None if value is None else float(value))
