"""Predeclared secondary whole-mission timing; primary Search metric unchanged."""
import math

def joint_completion_metric(rows,horizon,*,physics_dt=.1,stable_seconds=3.,connectivity_threshold=.9):
 if not rows or not all(math.isfinite(x) and x>0 for x in (horizon,physics_dt,stable_seconds)):raise ValueError('Positive finite horizon, sample interval and stability window required')
 previous=0.;since=None;first=None;safe=True
 for row in rows:
  t=float(row['time']);gap=t-previous
  if not math.isfinite(t) or gap<=0 or gap>physics_dt+1e-8:raise ValueError('Complete ordered physical-clock samples required')
  ratio=float(row['connectivity']);
  if not math.isfinite(ratio) or not 0<=ratio<=1:raise ValueError('Finite actual radio ratio required')
  safe=safe and row['collisions']==0 and row['no_fly_violations']==0
  eligible=(row['nonrelay_complete'] and ratio>=connectivity_threshold and not row['safety_active'] and row['escapes']==row['safety_records'])
  if not eligible:since=None
  elif since is None:since=t
  if first is None and since is not None and t-since>=stable_seconds-1e-8:first=t
  previous=t
 if abs(previous-horizon)>1e-8:raise ValueError('Full evaluation horizon required')
 final=rows[-1];safe=safe and not final['safety_active'] and final['escapes']==final['safety_records']
 success=safe and first is not None
 return dict(secondary_joint_completion_seconds=first if success else float(horizon),joint_success=bool(success),stable_seconds=stable_seconds,physics_dt=physics_dt,whole_window_safe=bool(safe),primary_search_metric_unchanged=True,acceptance_claim=False)
