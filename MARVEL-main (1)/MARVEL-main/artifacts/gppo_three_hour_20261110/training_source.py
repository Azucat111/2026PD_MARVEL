"""Compile the existing collector with only objective/trace instrumentation changed."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def replace_once(source,old,new):
 if source.count(old)!=1:raise ValueError('Changed training boundary: '+old[:80])
 return source.replace(old,new)

def source():
 s=(ROOT/'scripts/train_extended_gppo_batched.py').read_text()
 s=replace_once(s,'    validate_reward_protocol(cp,args.joint_rollout,args.reset_critic,args.relay_reward_weight)',"    assert args.joint_rollout and args.steps==300\n    assert cp['reward_protocol']=='joint_makespan_physical_h300_v2'")
 s=replace_once(s,'    reward_protocol="joint_physical_v1" if args.joint_rollout else "search_latency_v1"','    reward_protocol="joint_makespan_physical_h300_v2"')
 s=replace_once(s,'radio_samples=[];collision_times=[]','radio_samples=[];collision_times=[];joint_samples=[]')
 s=replace_once(s,'                obs,info=super().step(actions)','''                check_deadline()
                obs,info=super().step(actions)
                scheduler=self.high_level_scheduler
                joint_samples.append(dict(time=self.current_step*self.dt,connectivity=float(info['connectivity_ratio']),nonrelay_complete=self.tasks.all_complete(),safety_active=sorted(scheduler.safety.active_uav_ids),safety_records=len(scheduler.safety.records),escapes=scheduler.safety.successful_escapes,no_fly_violations=scheduler.safety.no_fly_violations,collisions=len(info['collisions'])))''')
 old="        returns=[latency_return(times,tr['time'],horizon)-penalty-args.relay_reward_weight*deficit for tr,deficit,penalty in zip(transitions,relay_deficits,penalties)]"
 s=replace_once(s,old,"""        reward_result=aligned_returns(joint_samples,[tr['time'] for tr in transitions])
        returns=reward_result['returns']
        (folder/'physical_joint_trace.json').write_text(json.dumps(joint_samples))
        (folder/'aligned_reward_proof.json').write_text(json.dumps(reward_result,indent=2))""")
 s=replace_once(s,"        dynamic=bool(args.balanced_domains and episode%4>=2)","        dynamic=bool(args.balanced_domains and episode%4>=2)")
 # Preserve the original balanced30/60 static/dynamic cycle.
 s=replace_once(s,"        'scripts/train_extended_gppo_heading.py'","        'scripts/train_extended_gppo_heading.py'") if "        'scripts/train_extended_gppo_heading.py'" in s else s
 s=replace_once(s,"training_stage='batched on-policy PPO true target latency'","training_stage='batched on-policy PPO full physical joint completion H300'")
 s=replace_once(s,"reward='negative remaining true capped detection latency / target_count / horizon; optional weighted actual radio deficit below .9 over full horizon; collision penalty'","reward='negative remaining physical joint completion seconds /300; complete H300 safety gate; failure capped300'")
 return s
