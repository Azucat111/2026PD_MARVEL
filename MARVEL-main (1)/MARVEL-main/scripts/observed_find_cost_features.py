"""Public-only features for a learned Find-time head over frozen GPPO representations."""
import numpy as np
FEATURE_PROTOCOL='extended-gppo-observed-find-cost-v1'
def find_cost_features(raw,edge9,ti,ui,current_time,public_hazards,dt,embedding):
 u=np.asarray(raw['raw_uav_features'],dtype=float);t=np.asarray(raw['raw_task_features'],dtype=float);e=np.asarray(raw['raw_edge_features'],dtype=float);mask=np.asarray(raw['mask'],dtype=bool);edge9=np.asarray(edge9,dtype=float)
 assert u.shape[1]==13 and t.shape[1]==12 and e.shape==(len(t),len(u),3) and edge9.shape==(len(t),len(u),9)
 if mask[ti,ui]:raise ValueError('Legal public action required')
 delta=t[ti,10:12]-u[ui,9:11];legal=e[:,:,2][~mask];row=e[ti,:,2][~mask[ti]];column=e[:,ui,2][~mask[:,ui]]
 context=np.array([len(u),len(t),float(current_time),len(legal),len(row),float(legal.min()),float(legal.mean()),float(row.min()),float(column.min())])
 hazard=[]
 for h in public_hazards:
  p=h.get('params',h);center=np.asarray(p['position'],dtype=float);spawn=float(h.get('spawn_step',0));step=float(current_time)/float(dt);active=step>=spawn
  radius=min(float(p['max_radius']),float(p['initial_radius'])+max(0.,step-spawn)*float(p['expansion_rate']))
  hazard.append([1.,float(active),float(np.linalg.norm(u[ui,9:11]-center))-radius,float(np.linalg.norm(t[ti,10:12]-center))-radius,radius,float(p['max_radius']),max(0.,spawn-step)*dt])
 risk=min(hazard,key=lambda h:h[2]) if hazard else [0.]*7
 values=np.concatenate([u[ui],t[ti],e[ti,ui],edge9[ti,ui],delta,[np.linalg.norm(delta)],context,risk,np.asarray(embedding,dtype=float)])
 if not np.isfinite(values).all():raise ValueError('Finite observed features required')
 return values
