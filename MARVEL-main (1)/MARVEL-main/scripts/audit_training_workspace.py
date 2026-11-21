"""Read-only workspace/CUDA/frozen-model audit; retain all user files."""
from pathlib import Path
import subprocess,json,hashlib,sys
import torch
root=Path(__file__).resolve().parents[1]
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
status=git('status','--porcelain').splitlines()
frozen=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
result=dict(repository_root=git('rev-parse','--show-toplevel'),branch=git('branch','--show-current'),head=git('rev-parse','HEAD'),
    worktree_dirty=bool(status),status_entries=len(status),cuda_available=torch.cuda.is_available(),
    cuda_device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    frozen_checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
    frozen_unchanged=hashlib.sha256(frozen.read_bytes()).hexdigest()=='57abf685f7a8226dcf71e1441333bf74da6e228e423f1bb0034e110975e3fcd9',
    user_scope='Training authorized; technical goals only; no PPT/report/commit/push/cleanup',status=status)
with (root/'artifacts/extended_training_workspace_audit_20261010.json').open('x') as handle:json.dump(result,handle,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='status'}),flush=True)
