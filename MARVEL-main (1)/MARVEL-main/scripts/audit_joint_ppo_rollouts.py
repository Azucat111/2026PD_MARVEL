"""Read-only audit of captured real PPO behavior probabilities and physical rewards."""
from pathlib import Path
import sys,json,argparse
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from integrations.gppo.gppo_model import GPPOActorCritic
from gppo_rollout_batch import model_state_fingerprint
from gppo_joint_reward import remaining_connectivity_deficit
from train_extended_gppo_batched import latency_return

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--episodes',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    folder=ROOT/a.folder;manifest=json.loads((folder/'manifest.json').read_text());history=json.loads((folder/'progress.json').read_text())[:a.episodes]
    assert len(history)==a.episodes
    torch.set_num_threads(1);worst_log=0.;worst_value=0.;worst_return=0.;count=0;fingerprints=set()
    for row in history:
        cp=torch.load(folder/row['behavior_checkpoint'],map_location='cpu',weights_only=False)
        model=GPPOActorCritic(edge_feature_dim=6);model.load_state_dict(cp['model'],strict=True);model.eval()
        fingerprint=model_state_fingerprint(model);assert fingerprint==row['behavior_sha256'];fingerprints.add(fingerprint)
        episode=folder/f"episode_{row['episode']:03d}"
        records=json.loads((episode/'actor_inputs.json').read_text());radio=json.loads((episode/'physical_radio_reward_trace.json').read_text());collisions=json.loads((episode/'physical_collision_reward_times.json').read_text())
        assert len(records)==len(row['returns'])
        for record,expected_return in zip(records,row['returns']):
            batch={key:torch.as_tensor(value) for key,value in record['batch'].items()}
            with torch.no_grad():output=model(**batch)
            action=record['action'];assert not bool(batch['action_mask'].reshape(-1)[action])
            dist=torch.distributions.Categorical(logits=output['flat_logits'][0]/manifest['args']['temperature'])
            worst_log=max(worst_log,abs(float(dist.log_prob(torch.tensor(action)))-record['log_prob']))
            worst_value=max(worst_value,abs(float(output['value'][0])-record['value']))
            time=record['time'];horizon=row['metric']['horizon_seconds']
            observed=latency_return(row['metric']['capped_detection_seconds'],time,horizon)-.1*sum(t>time for t in collisions)-manifest['args']['relay_reward_weight']*remaining_connectivity_deficit(radio,time,horizon)
            worst_return=max(worst_return,abs(observed-expected_return));count+=1
    assert worst_log<1e-5 and worst_value<1e-5 and worst_return<1e-10
    result=dict(episodes=len(history),transitions=count,behavior_fingerprints=sorted(fingerprints),max_old_log_probability_error=worst_log,max_old_value_error=worst_value,max_physical_reward_error=worst_return,valid_actions=True,passed=True,acceptance_claim=False)
    with (ROOT/a.out).open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
