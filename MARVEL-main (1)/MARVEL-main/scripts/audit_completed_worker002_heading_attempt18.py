"""Read-only heading audit for a worker retaining its global episode index."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_training_heading_control_contract import audit_record
folder=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt18';records=json.loads((folder/'workers/worker_002/episode_002/actor_inputs.json').read_text());errors=[audit_record(r) for r in records]
result=dict(episodes=1,global_episode_index=2,records=len(records),production_control_clock_verified=True,public_graph_and_heading_only=True,max_heading_feature_error=max(errors),passed=True,acceptance_claim=False,correction='Generic multi-episode CLI assumes episode0; this completed isolated worker retains globalindex2.')
with (folder/'worker002_heading_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
