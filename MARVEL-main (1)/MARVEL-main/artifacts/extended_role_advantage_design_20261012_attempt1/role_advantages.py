"""Training-only relative outcome normalization; no reward or input modification."""
import torch

def normalize_role_advantages(advantages,roles):
    if not isinstance(advantages,torch.Tensor) or advantages.ndim!=1 or not advantages.is_floating_point():
        raise ValueError('Need one-dimensional floating advantages')
    if len(roles)!=len(advantages) or not torch.isfinite(advantages).all():
        raise ValueError('Role/value contract mismatch or nonfinite advantage')
    if any(not isinstance(r,int) or isinstance(r,bool) for r in roles):
        raise ValueError('Integer role IDs required')
    values=advantages.detach();result=torch.zeros_like(values)
    for role in sorted(set(roles)):
        indices=[i for i,r in enumerate(roles) if r==role]
        group=values[indices];centered=group-group.mean();scale=group.std(unbiased=False)
        if len(indices)>1 and float(scale)>1e-8:
            result[indices]=centered/scale
    return result
