"""Paired natural rollout of an independent trained candidate.
Same physical runtime, profiles, legal masks and rule baseline as the frozen test.
Experimental model is installed in this process only, after standard setup.
"""
from pathlib import Path
import argparse,json,hashlib,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from train_extended_gppo import PROTOCOL

def install_model(adapter,path):
    candidate=torch.load(path,map_location=adapter.device,weights_only=False)
    if candidate.get('protocol_version')!=PROTOCOL:raise ValueError('Wrong experimental checkpoint protocol')
    adapter.model.load_state_dict(candidate['model'],strict=True)
    adapter.model.eval()
    return candidate

def main():
    import benchmark_fair_search as bench
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--uavs',type=int,choices=[30,60],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=80);args=p.parse_args()
    candidate=Path(args.checkpoint).resolve();candidate_hash=hashlib.sha256(candidate.read_bytes()).hexdigest()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    base=bench.MARVELPolicyAdapter
    class CandidateAdapter(base):
        def setup(self):
            result=super().setup()
            install_model(self.scheduler.allocator.adapter,candidate)
            return result
    bench.MARVELPolicyAdapter=CandidateAdapter
    paths=[ROOT/'scripts/evaluate_extended_gppo.py',ROOT/'scripts/benchmark_fair_search.py',
           ROOT/'scripts/train_extended_gppo.py']
    # Snapshot production Python sources too; both arms must use identical code.
    paths+=list((ROOT/'integrations').rglob('*.py'))+list((ROOT/'utils').rglob('*.py'))
    hashes={str(file.relative_to(ROOT)):hashlib.sha256(file.read_bytes()).hexdigest() for file in paths}
    frozen_hash=hashlib.sha256(bench.CHECKPOINT.read_bytes()).hexdigest()
    manifest=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seed=args.seed,uavs=args.uavs,checkpoint=str(candidate),checkpoint_sha256=candidate_hash,
        protocol=PROTOCOL,frozen_checkpoint_sha256=frozen_hash,source_hashes=hashes,
        comparison='Only GPPO actor weights change relative to rule; runtime/config/profile/masks shared',
        stage='development rollout, not final 15% acceptance',horizon_seconds=args.steps)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        results=[bench.run(out,arm,args.uavs,args.seed,args.steps,physical_coverage=True,stop_on_completion=True)
                 for arm in ('gppo','rule')]
    finally:bench.MARVELPolicyAdapter=base
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==candidate_hash
    assert hashlib.sha256(bench.CHECKPOINT.read_bytes()).hexdigest()==frozen_hash
    assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest for rel,digest in hashes.items())
    assert json.loads((out/'gppo/initial.json').read_text())==json.loads((out/'rule/initial.json').read_text())
    (out/'paired.json').write_text(json.dumps(results,indent=2))
    gm=results[0]['metric']['mean_capped_seconds'];rm=results[1]['metric']['mean_capped_seconds']
    (out/'comparison.json').write_text(json.dumps(dict(valid=all(r['valid_for_comparison'] for r in results),
        efficiency_gain=rm/gm-1 if gm else None,acceptance_claim=False),indent=2))

if __name__=='__main__':main()
