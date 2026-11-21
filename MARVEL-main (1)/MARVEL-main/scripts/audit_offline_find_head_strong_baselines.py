"""Postfit baseline audit only; candidate remains unchanged."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_offline_find_cost_head_20261015_attempt28_retry01'
a=np.load(OUT/'observed_training_arrays.npz');x=a['features'];y=a['delay'];hold=a['holdout']
p=OUT/'candidate.joblib';sha=hashlib.sha256(p.read_bytes()).hexdigest();artifact=joblib.load(p);model=artifact['estimator']
def metrics(pred):return dict(mae=float(mean_absolute_error(y[hold],pred)),rmse=float(mean_squared_error(y[hold],pred)**.5))
# Layout is defined explicitly by observed_find_cost_features: 13+12+3+9+3+9+7+128.
assert x.shape[1]==184
sensor_eta=x[:,34]*80.
columns=[25,27,31,32,34,39,40,42]
linear=make_pipeline(StandardScaler(),Ridge(alpha=10.));linear.fit(x[~hold][:,columns],y[~hold])
usage=np.zeros(x.shape[1],dtype=int)
for group in model._predictors:
 for tree in group:
  for node in tree.nodes:
   if not node['is_leaf']:usage[int(node['feature_idx'])]+=1
report=dict(candidate_unchanged=True,candidate_sha256=sha,heldout_prediction=metrics(model.predict(x[hold])),nominal_sensor_eta=metrics(sensor_eta[hold]),training_only_ridge_baseline=metrics(linear.predict(x[hold][:,columns])),graph_representation_split_count=int(usage[56:].sum()),raw_public_split_count=int(usage[:56].sum()),feature_count=184,new_simulations=0,acceptance_claim=False)
(OUT/'strong_baseline_audit.json').write_text(json.dumps(report,indent=2));assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;print(json.dumps(report),flush=True)
