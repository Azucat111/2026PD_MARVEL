"""Training-only bounds from real sensor events; never actor inputs."""
import math

def latency_interval(metric,final_horizon=300.):
 n=int(metric['total_targets']);found=int(metric['detected_targets']);h=float(metric['horizon_seconds']);times=list(map(float,metric['capped_detection_seconds']))
 if n<=0 or len(times)!=n or not 0<=found<=n or not 0<h<=final_horizon or any(not math.isfinite(t) or t<0 or t>h for t in times):raise ValueError('Invalid capped physical sensor metric')
 missing=n-found
 if sum(t==h for t in times)<missing:raise ValueError('Missing target lacks its declared censor time')
 lower=math.fsum(times)/n;upper=(math.fsum(times)+missing*(final_horizon-h))/n
 return dict(lower=lower,upper=upper,exact=missing==0,found=found,targets=n)

def efficiency_interval(baselines,candidates):
 if not baselines or len(baselines)!=len(candidates):raise ValueError('Paired cases required')
 for r in baselines+candidates:
  if not 0<=r['lower']<=r['upper'] or not math.isfinite(r['upper']):raise ValueError('Finite latency bounds required')
 cl=sum(r['lower'] for r in candidates);cu=sum(r['upper'] for r in candidates)
 if cl<=0 or cu<=0:raise ValueError('Positive candidate latency required')
 return dict(lower=sum(r['lower'] for r in baselines)/cu-1,upper=sum(r['upper'] for r in baselines)/cl-1,acceptance_claim=False)
