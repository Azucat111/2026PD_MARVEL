"""Exercise the existing GAE primitive against saved physical rollout returns."""
from pathlib import Path
import sys,ast,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
upstream=Path('/home/nick/MARVEL/utils/gppo_ppo.py');source=upstream.read_text();tree=ast.parse(source)
f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_gae')
from typing import Sequence,Tuple
ns=dict(np=np,Sequence=Sequence,Tuple=Tuple)
exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(upstream),'exec'),ns)
folder=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19';history=json.loads((folder/'progress.json').read_text())
error=0.;count=0
for ep in history:
 records=json.loads((folder/f"episode_{ep['episode']:03d}"/'actor_inputs.json').read_text())
 returns=np.asarray(ep['returns'],dtype=np.float64);values=[r['value'] for r in records]
 # Suffix targets telescope into incremental decision rewards; this is an identity check,
 # not evidence that standard gamma/lambda or a new learner will perform better.
 increments=[float(returns[i]-(returns[i+1] if i+1<len(returns) else 0.)) for i in range(len(returns))]
 advantages,reconstructed=ns['compute_gae'](increments,values,[False]*(len(values)-1)+[True],gamma=1.,gae_lambda=1.,last_value=0.)
 error=max(error,float(np.abs(reconstructed-returns).max()));count+=len(values)
assert count==1267 and error<1e-6
result=dict(passed=True,episodes=len(history),transitions=count,max_return_error=error,reused_function='compute_gae',source=str(upstream),source_sha256=hashlib.sha256(upstream.read_bytes()).hexdigest(),gamma=1.,gae_lambda=1.,preserves_existing_undiscounted_objective=True,training_steps=0,acceptance_claim=False,limits='Identity only; generic gamma=.99/lambda=.95 would change targets; no claim of solved credit assignment')
(ROOT/'artifacts/extended_system_review_20261014_attempt19/existing_gae_reuse_contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
