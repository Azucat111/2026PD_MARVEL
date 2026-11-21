from pathlib import Path
import json,torch,sys
R=Path.cwd();O=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(R/'scripts')]
from gppo_rollout_batch import model_state_fingerprint
from integrations.gppo.gppo_model import GPPOActorCritic
paths={'rejected':O/'update/checkpoint_batch_000.pth','guarded':O/'guarded/update/checkpoint_batch_000.pth'};rows={}
for name,p in paths.items():
 cp=torch.load(p,map_location='cpu',weights_only=True);steps=sorted(set(float(s['step']) for s in cp['optimizer']['state'].values() if 'step' in s));m=GPPOActorCritic(edge_feature_dim=6);m.load_state_dict(cp['model']);rows[name]=dict(adam_steps=steps,model_fingerprint=model_state_fingerprint(m))
G=json.loads((O/'guarded/update_proof.json').read_text());assert rows['rejected']['adam_steps']==[2.] and rows['guarded']['adam_steps']==[1.];assert rows['guarded']['model_fingerprint']==G['epoch_transactions'][0]['retained_model_fingerprint'];assert G['epoch_transactions'][1]['matches_original_rejected_model'];(O/'guarded/actual_saved_state_verification.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))

