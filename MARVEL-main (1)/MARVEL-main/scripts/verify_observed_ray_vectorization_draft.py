from pathlib import Path
import sys,importlib.util,json,time,statistics
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import utils.geometry as geometry
import integrations.gppo.search_recovery as recovery
oldFrame=geometry.GeometryFrame;oldRecovery=recovery.ObservedSearchRecovery
folder=ROOT/'artifacts/extended_observed_ray_vectorization_20261011_attempt1'
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,folder/file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
newFrame=load('ray_draft_geometry','geometry_draft.py').GeometryFrame
newRecovery=load('ray_draft_recovery','search_recovery_draft.py').ObservedSearchRecovery
geometry.GeometryFrame=newFrame;recovery.ObservedSearchRecovery=newRecovery
code=pytest.main(['-q',str(ROOT/'platform_tests/test_vector_observed_ray.py')]);assert code==0
rng=np.random.default_rng(4360);comparisons=0;segment_comparisons=0;benchmark=[]
for ox,oy,size,w,h in [(0.,0.,1.,80,80),(-3.2,-7.6,.4,50,60),(7.,-11.,1.,40,30)]:
 old=oldRecovery(oldFrame(ox,oy,size,w,h));new=newRecovery(newFrame(ox,oy,size,w,h))
 for i in range(3340):
  a=np.array([ox,oy])+rng.uniform([0,0],[w*size,h*size])
  if i%20==0:b=np.array([ox,oy])+rng.uniform([0,0],[w*size,h*size])
  else:b=a+rng.uniform(-1.5*size,1.5*size,2)
  if i%17==0:a=np.round((a-[ox,oy])/size)*size+[ox,oy]
  if i%19==0:b=np.nextafter(a,np.inf)
  if i%23==0:b=a.copy()
  assert list(old._ray_cells(a,b))==list(new._ray_cells(a,b)),(a,b)
  comparisons+=1
  if i%10==0:benchmark.append((old,new,a,b))
 old.cells={(c,r):int(rng.random()>.75) for c in range(w) for r in range(h) if rng.random()>.1};new.cells=dict(old.cells)
 for i in range(300):
  a=np.array([ox,oy])+rng.uniform([0,0],[w*size,h*size]);b=a+rng.uniform(-2,2,2)
  assert old.segment_safe(a,b)==new.segment_safe(a,b);segment_comparisons+=1
measurements=[]
for _ in range(3):
 start=time.process_time()
 for old,new,a,b in benchmark:list(old._ray_cells(a,b))
 old_time=time.process_time()-start;start=time.process_time()
 for old,new,a,b in benchmark:list(new._ray_cells(a,b))
 new_time=time.process_time()-start;measurements.append(dict(old_cpu_seconds=old_time,new_cpu_seconds=new_time,speedup=old_time/new_time))
result=dict(passed=True,tests=4,exact_ray_sequences= comparisons,exact_segment_results=segment_comparisons,measurements=measurements,median_ray_cpu_speedup=statistics.median(m['speedup'] for m in measurements),scope='Offline observed ray computation only; no environment rerun and no whole-runtime speedup claim',production_not_yet_installed=True)
with (folder/'staged_equivalence.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
