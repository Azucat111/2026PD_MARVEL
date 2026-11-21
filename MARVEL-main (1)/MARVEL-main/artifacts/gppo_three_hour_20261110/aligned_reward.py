"""Training-only prototype. Reuses the frozen evaluation metric, never actor input.

No policy is trained on replayed evaluation data. H300 is mandatory; changing
this reward requires explicit critic and optimizer reset in a future trainer.
"""
import math
from gppo_joint_completion_metric import joint_completion_metric

PROTOCOL='joint_makespan_physical_h300_v2'
HORIZON=300.

def aligned_returns(rows,decision_times):
 times=[float(t) for t in decision_times]
 if any(not math.isfinite(t) or not 0<=t<=HORIZON for t in times):
  raise ValueError('Decision times must be finite and within H300')
 metric=joint_completion_metric(rows,HORIZON,physics_dt=.1,stable_seconds=3.,connectivity_threshold=.9)
 completion=float(metric['secondary_joint_completion_seconds'])
 return dict(protocol=PROTOCOL,completion_seconds=completion,success=metric['joint_success'],
  whole_window_safe=metric['whole_window_safe'],returns=[-max(0.,completion-t)/HORIZON for t in times],
  horizon_seconds=HORIZON,metric_reused=True,actor_input=False,requires_fresh_on_policy_rollouts=True,
  requires_critic_and_optimizer_reset=True)
