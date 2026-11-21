"""Training-only wall-time wrapper; unchanged physical horizon and PPO implementation."""
from pathlib import Path
import functools,math,json,hashlib,sys,argparse
ROOT=Path(__file__).resolve().parents[1]
def bind_wall_budget(run,budget):
    if not math.isfinite(budget) or budget<=0:raise ValueError('Positive finite wall budget required')
    @functools.wraps(run)
    def wrapped(*args,**kwargs):
        if 'wall_budget_seconds' in kwargs and kwargs['wall_budget_seconds']!=budget:raise ValueError('Conflicting wall budget')
        kwargs['wall_budget_seconds']=budget
        return run(*args,**kwargs)
    return wrapped

def main():
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--training-wall-budget-seconds',type=float,required=True)
    wrapper_args,remaining=parser.parse_known_args()
    out_arg=remaining[remaining.index('--out')+1]
    old=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261011_attempt11.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in old['source_hashes'].items())
    proof=dict(wall_budget_seconds=wrapper_args.training_wall_budget_seconds,physical_semantics_unchanged=True,ppo_implementation_unchanged=True,wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_hashes=old['source_hashes'])
    with (ROOT/(out_arg+'_execution_manifest.json')).open('x') as f:json.dump(proof,f,indent=2)
    import benchmark_fair_search as bench
    import train_extended_gppo_batched as trainer
    original=bench.run;bench.run=bind_wall_budget(original,wrapper_args.training_wall_budget_seconds)
    sys.argv=[str(ROOT/'scripts/train_extended_gppo_batched.py'),*remaining]
    try:trainer.main()
    finally:bench.run=original
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==proof['wrapper_sha256']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in proof['source_hashes'].items())
if __name__=='__main__':main()
