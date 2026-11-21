from pathlib import Path
import sys,json,subprocess,hashlib
import torch
ROOT=Path(__file__).resolve().parents[1]
get=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
r=dict(repository_root=get('rev-parse','--show-toplevel'),branch=get('branch','--show-current'),head=get('rev-parse','HEAD'),status=get('status','--short'),cuda_available=torch.cuda.is_available(),torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,frozen_sha256=hashlib.sha256(Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth').read_bytes()).hexdigest(),source_installed=True,acceptance_claim=False)
assert r['branch']=='extended-scalability' and r['head']=='5143fbdc4bdadf907f59c433e81ab11b2e98c6dc'
assert r['frozen_sha256']=='57abf685f7a8226dcf71e1441333bf74da6e228e423f1bb0034e110975e3fcd9'
with (ROOT/'artifacts/extended_integrated_candidate_design_20261011_attempt1/environment.json').open('x') as s:json.dump(r,s,indent=2)
print(json.dumps({k:v for k,v in r.items() if k!='status'}))
