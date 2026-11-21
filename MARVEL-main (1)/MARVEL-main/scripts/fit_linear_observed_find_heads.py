"""Compare two fixed low-cost linear heads against nominal sensor ETA; no simulation."""
from pathlib import Path
import json,hashlib,time
import numpy as np,joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'artifacts/extended_offline_find_cost_head_20261015_attempt28_retry01';OUT=ROOT/'artifacts/extended_linear_find_cost_preflight_20261015_attempt29';OUT.mkdir(exist_ok=False)
plan=dict(training_data='Same917 strict own-Find labels and predeclared episode blocks',heads=['linear_public8_alpha10','linear_gppo184_alpha100'],models_fitted_before_scoring=True,independent_acceptance_claim=False,goal_complete=False,expensive_training_allowed=False)
(OUT/'plan.json').write_text(json.dumps(plan,indent=2))
a=np.load(OLD/'observed_training_arrays.npz');x=a['features'];y=a['delay'];hold=a['holdout'];base=joblib.load(OLD/'candidate.joblib');columns=[25,27,31,32,34,39,40,42]
models={'linear_public8_alpha10':make_pipeline(ColumnTransformer([('public',StandardScaler(),columns)],remainder='drop'),Ridge(alpha=10.)),'linear_gppo184_alpha100':make_pipeline(StandardScaler(),Ridge(alpha=100.))}
frozen=[]
for name,model in models.items():
 started=time.perf_counter();model.fit(x[~hold],y[~hold]);seconds=time.perf_counter()-started
 artifact={**base,'estimator':model,'calibration_method':name,'training_scope':'Same original726 training records; original191 regression validation labels are model-selection data now, not independent acceptance','fit_seconds':seconds,'requires_graph_embedding':name=='linear_gppo184_alpha100'}
 p=OUT/f'{name}.joblib';joblib.dump(artifact,p);frozen.append(dict(name=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fit_seconds=seconds))
def scores(pred):return dict(mae=float(mean_absolute_error(y[hold],pred)),rmse=float(mean_squared_error(y[hold],pred)**.5))
results=[]
for item in frozen:results.append(dict(**item,validation=scores(models[item['name']].predict(x[hold]))))
report=dict(models=results,nominal_sensor_eta=scores(x[hold,34]*80),old_gboost=scores(base['estimator'].predict(x[hold])),new_simulations=0,independent_acceptance_claim=False,goal_complete=False,expensive_training_allowed=False)
(OUT/'offline_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
