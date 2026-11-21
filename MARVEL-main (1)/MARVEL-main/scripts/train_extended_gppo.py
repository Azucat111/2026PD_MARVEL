"""Independent observation-only 30/60-UAV GPPO warm start.
Rule imitation is a training stage, never evidence of a 15% efficiency gain.
Does not change production loaders or overwrite the frozen checkpoint.
"""
from pathlib import Path
import argparse, json, hashlib, random, sys, os, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from integrations.gppo.gppo_model import GPPOActorCritic

PROTOCOL='extended-gppo-observed-warmstart-v1'

def validate_seed_split(train,dev,final):
    groups=[set(train),set(dev),set(final)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Training, development and final seeds must be disjoint')

def example_from_record(record,device):
    if record.get('candidate_budget',0):raise ValueError('Filtered candidate graph excluded')
    uids=record['uav_ids'];tids=record['task_ids'];nt=len(tids);nu=len(uids)
    if not nt or not nu:return None
    if len(set(uids))!=nu or len(set(tids))!=nt:raise ValueError('Duplicate IDs')
    mask=np.asarray(record['mask'],dtype=bool)
    rawtask=np.asarray(record['raw_task_features'],dtype=np.float32)
    rawedge=np.asarray(record['raw_edge_features'],dtype=np.float32)
    if mask.shape!=(nt,nu) or rawtask.shape!=(nt,12) or rawedge.shape!=(nt,nu,3):
        raise ValueError('Malformed raw observation dimensions')
    arrays={
        'uav_features':(record['model_uav_features'],(nu,13),torch.float32),
        'task_features':(record['model_task_features'],(nt,12),torch.float32),
        'edge_features':(record['model_edge_features'],(nt,nu,3),torch.float32),
        'capability_matrix':(record['capability_matrix'],(nt,nu),torch.bool),
        'precedence_matrix':(record['precedence_matrix'],(nt,nt),torch.bool),
        'action_mask':(record['mask'],(nt,nu),torch.bool)}
    batch={}
    for key,(value,shape,dtype) in arrays.items():
        array=np.asarray(value)
        if array.shape!=shape or not np.isfinite(array).all():raise ValueError('Invalid '+key)
        batch[key]=torch.as_tensor(array,dtype=dtype,device=device).unsqueeze(0)
    if not np.isfinite(rawtask).all() or not np.isfinite(rawedge).all():raise ValueError('Nonfinite raw features')
    candidates=[(-float(rawtask[t,0]),float(rawedge[t,u,2]),int(tids[t]),int(uids[u]),t,u)
                for t,u in np.argwhere(~mask)]
    if not candidates:return None
    *_,t,u=min(candidates)
    return dict(batch=batch,label=int(t*nu+u),costs=rawedge[:,:,2],priority=rawtask[:,0],mask=mask,
                uav_ids=uids,task_ids=tids)

def train_step(model,optimizer,ex):
    optimizer.zero_grad(set_to_none=True)
    logits=model(**ex['batch'])['flat_logits']
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([ex['label']],device=logits.device))
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return float(loss.detach())

@torch.no_grad()
def evaluate(model,examples):
    model.eval();correct=0;regrets=[];priority_errors=0
    for ex in examples:
        action=int(model(**ex['batch'])['flat_logits'].argmax().item())
        nt,nu=ex['mask'].shape;t,u=divmod(action,nu);lt,lu=divmod(ex['label'],nu)
        if ex['mask'][t,u]:raise AssertionError('Model selected invalid pair')
        correct+=action==ex['label'];priority_errors+=ex['priority'][t]<ex['priority'][lt]
        regrets.append(float(ex['costs'][t,u]-ex['costs'][lt,lu]))
    return dict(graphs=len(examples),teacher_accuracy=correct/max(1,len(examples)),
                priority_errors=int(priority_errors),mean_observed_cost_regret=float(np.mean(regrets)) if regrets else None,
                caveat='Observed estimates only; not true rollout efficiency')

def selection_key(metrics):
    # Priority violations always dominate cost, regardless of cost magnitude.
    return (int(metrics['priority_errors']),float(metrics['mean_observed_cost_regret']))

def load_examples(seeds,device):
    examples=[];sources=[]
    for folder in sorted((ROOT/'artifacts').iterdir()):
        if not folder.is_dir() or not (folder/'paired.json').exists():continue
        pair=json.loads((folder/'paired.json').read_text())
        if not isinstance(pair,list) or not pair:continue
        if pair[0].get('seed') not in seeds or pair[0].get('uavs') not in (30,60):continue
        if not all(p.get('valid_for_comparison') for p in pair):continue
        for arm in ('gppo','rule'):
            file=folder/arm/'decisions.json'
            if not file.exists():continue
            source_hash=hashlib.sha256(file.read_bytes()).hexdigest();used=0
            for record in json.loads(file.read_text()):
                if 'model_uav_features' not in record or record.get('candidate_budget',0):continue
                ex=example_from_record(record,device)
                if ex is not None:examples.append(ex);used+=1
            sources.append(dict(path=str(file),sha256=source_hash,seed=pair[0]['seed'],graphs=used))
    # Exact duplicate public graphs are not extra evidence or extra training weight.
    unique={}
    for ex in examples:
        digest=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in ex['batch'].values())).hexdigest()
        unique.setdefault(digest,ex)
    return list(unique.values()),sources

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--train-seeds',type=int,nargs='+',default=[56789,67890]);p.add_argument('--dev-seeds',type=int,nargs='+',default=[78901])
    p.add_argument('--final-seeds',type=int,nargs='+',default=[101101,202202,303303]);p.add_argument('--seed',type=int,default=9147)
    p.add_argument('--lr',type=float,default=.0001);p.add_argument('--device',default='cuda');args=p.parse_args()
    validate_seed_split(args.train_seeds,args.dev_seeds,args.final_seeds)
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    initial=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
    initial_hash=hashlib.sha256(initial.read_bytes()).hexdigest()
    checkpoint=torch.load(initial,map_location='cpu',weights_only=False)
    model=GPPOActorCritic().to(args.device);model.load_state_dict(checkpoint['model'],strict=True)
    # eval mode fixes RReLU inference slope; gradients remain enabled.
    model.eval();optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    train,train_sources=load_examples(set(args.train_seeds),args.device)
    dev,dev_sources=load_examples(set(args.dev_seeds),args.device)
    if not train or not dev:raise RuntimeError('Need nonempty independent train/dev observations')
    manifest=dict(protocol=PROTOCOL,args=vars(args),initial_checkpoint=str(initial),initial_sha256=initial_hash,
        train_sources=train_sources,dev_sources=dev_sources,train_graphs=len(train),dev_graphs=len(dev),
        device=str(args.device),cuda_device=torch.cuda.get_device_name() if args.device=='cuda' else None,
        actor_information='Saved public normalized graph and legal masks only; no map or hidden target reads',
        final_seeds_reserved=args.final_seeds,stage='Rule imitation warm start; not final acceptance',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    baseline=dict(train=evaluate(model,train),dev=evaluate(model,dev));best=(float('inf'),float('inf'));history=[];started=time.monotonic()
    for epoch in range(args.epochs):
        random.shuffle(train);loss=sum(train_step(model,optimizer,ex) for ex in train)/len(train)
        if epoch%10==0 or epoch==args.epochs-1:
            metrics=evaluate(model,dev);score=selection_key(metrics)
            row=dict(epoch=epoch+1,loss=loss,train=evaluate(model,train),dev=metrics,seconds=time.monotonic()-started)
            history.append(row);print(json.dumps(row),flush=True)
            if score<best:
                best=score;torch.save(dict(protocol_version=PROTOCOL,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    epoch=epoch+1,initial_sha256=initial_hash,validation=row),out/'best.pth')
            (out/'progress.json').write_text(json.dumps(dict(baseline=baseline,history=history),indent=2))
    assert hashlib.sha256(initial.read_bytes()).hexdigest()==initial_hash
    torch.save(dict(protocol_version=PROTOCOL,model=model.state_dict(),epoch=args.epochs,initial_sha256=initial_hash),out/'last.pth')
    (out/'complete.json').write_text(json.dumps(dict(initial_unchanged=True,seconds=time.monotonic()-started,best_dev_score=best),indent=2))

if __name__=='__main__':main()
