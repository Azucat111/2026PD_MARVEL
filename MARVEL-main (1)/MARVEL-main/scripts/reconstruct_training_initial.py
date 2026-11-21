"""Reconstruct audited weight-only initial model after explicit critic reset."""
from pathlib import Path
import argparse,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from integrations.gppo.gppo_model import GPPOActorCritic
from train_extended_gppo_heading import reset_value_head

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 manifest=json.loads((ROOT/a.run/'manifest.json').read_text());initial=Path(manifest['args']['initial'])
 if not initial.is_absolute():initial=ROOT/initial
 if hashlib.sha256(initial.read_bytes()).hexdigest()!=manifest['initial_sha256']:raise ValueError('Initial checksum mismatch')
 if not manifest['args'].get('reset_critic'):raise ValueError('Only explicit critic reset reconstruction supported')
 cp=torch.load(initial,map_location='cpu',weights_only=False)
 model=GPPOActorCritic(edge_feature_dim=6);model.load_state_dict(cp['model'],strict=True);reset_value_head(model)
 out=ROOT/a.out
 if out.exists():raise FileExistsError(out)
 torch.save(dict(protocol_version=cp['protocol_version'],model=model.state_dict(),reconstructed=True,initial_sha256=manifest['initial_sha256'],transform='Explicit zero value output; actor/encoder unchanged; weight-only reconstruction'),out)
 print(str(out))
if __name__=='__main__':main()
