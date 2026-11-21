"""Explicit audited extension of the existing physical rollout trainer."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def training_source():
 source=(ROOT/'scripts/train_extended_gppo_batched.py').read_text()
 replacements={
 'from integrations.gppo.heading_features import HEADING_PROTOCOL as PROTOCOL,HeadingObservations,heading_batch':'from integrations.gppo.heading_features import HeadingObservations\nfrom integrations.gppo.opportunity_features import OPPORTUNITY_PROTOCOL as PROTOCOL,opportunity_batch as heading_batch,public_search_context',
 'GPPOActorCritic(edge_feature_dim=6)':'GPPOActorCritic(edge_feature_dim=9)',
 "batch=heading_batch(graph,self.heading_observations,'cuda')":"self.heading_observations.public_search_context=public_search_context(self.scheduler)\n                    batch=heading_batch(graph,self.heading_observations,'cuda')",
 "audit_heading_control_record(transitions[-1]['record'])":"from audit_opportunity_inputs import capture_public_snapshot,audit_record\n                    transitions[-1]['record']['public_opportunity_snapshot']=capture_public_snapshot(graph,self.heading_observations.public_search_context)\n                    audit_record(transitions[-1]['record'])",
 }
 for old,new in replacements.items():
  if source.count(old)!=1:raise ValueError('Ambiguous training extension boundary '+old)
  source=source.replace(old,new)
 return source
