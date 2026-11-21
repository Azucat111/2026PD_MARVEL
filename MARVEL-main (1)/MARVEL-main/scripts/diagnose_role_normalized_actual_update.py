"""Offline diagnostic update using actual last frozen batch; not an acceptance checkpoint."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
STAGE=ROOT/'artifacts/extended_role_advantage_design_20261012_attempt1';sys.path.insert(0,str(STAGE))
import torch
from gppo_parallel_rollouts import load_captured_episode,existing_update_function,model_state_fingerprint,validate_frozen_rows
from gppo_rollout_batch import FrozenBehaviorBatch
from integrations.gppo.gppo_model import GPPOActorCritic
F=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt15';manifest=json.loads((F/'manifest.json').read_text());args=SimpleNamespace(**manifest['args'])
cp=torch.load(F/'behavior_batch_007.pth',map_location='cpu',weights_only=False);model=GPPOActorCritic(edge_feature_dim=6);model.load_state_dict(cp['model']);model.eval();torch.set_num_threads(1)
optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);optimizer.load_state_dict(cp['optimizer'])
collector=FrozenBehaviorBatch(8);collector.begin(model);rows=[]
for i in range(56,64):
 row,tr=load_captured_episode(F,i,'cpu');collector.add_episode(model,tr,row['returns']);rows.append(row)
validate_frozen_rows(rows,collector.policy_sha256)
source=(ROOT/'scripts/train_extended_gppo_batched.py').read_text();needle='advantages=(targets-old_values).detach()';assert source.count(needle)==1
source=source.replace(needle,'from role_advantages import normalize_role_advantages\n            advantages=normalize_role_advantages((targets-old_values).detach(),[tr["role"] for tr in train])')
source=source.replace("device='cuda'", "device='cpu'")
update=existing_update_function(source);updates=[];out=STAGE/'actual_last_batch_replay_attempt2';out.mkdir(exist_ok=False)
update(model,optimizer,collector,args,out,63,updates,manifest['initial_sha256'],manifest['protocol'],manifest['reward_protocol'])
result=dict(episodes=8,transitions=sum(u['transitions'] for u in updates),only_change='Update-time within-role normalization of detached true-return advantages',normalized_update=updates,production_source_unchanged=True,scope='Stored actual frozen on-policy batch replay only. Diagnostic checkpoint is not a candidate and must not be used in independent acceptance or new continuation.',acceptance_claim=False)
(out/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
