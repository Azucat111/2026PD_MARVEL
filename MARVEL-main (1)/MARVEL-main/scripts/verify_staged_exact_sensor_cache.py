"""Read-only staged sensor equality and CPU benchmark; not a policy efficiency claim."""
from pathlib import Path
import sys,json,time,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));STAGE=ROOT/'artifacts/extended_exact_sensor_cache_design_20261012_attempt1'
from utils.sensor_models import IdealSensor
from utils.geometry import GeometryFrame
spec=importlib.util.spec_from_file_location('utils.sensor_models_cache_draft',STAGE/'sensor_models_draft.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rng=np.random.default_rng(62800);grid=(rng.random((40,40))<.1).astype(np.uint8)
frames=[GeometryFrame.extended(width_cells=40,height_cells=40),GeometryFrame(origin_x=-3.,origin_y=1.5,cell_size=.4,width_cells=40,height_cells=40)]
checked=0
for frame in frames:
 params=dict(fov=120.,range=6.,los_occlusion=True);old=IdealSensor(params,frame=frame);new=module.IdealSensor(dict(**params,visible_cells_cache_entries=128),frame=frame)
 for k in range(160):
  p=np.array(frame.origin)+rng.uniform(0,40,2)*frame.cell_size;h=float(rng.uniform(0,360))
  if k%10==0:grid[k%40,(k*3)%40]^=1
  a=old._visible_cells(p,h,grid);b=new._visible_cells(p,h,grid);c=new._visible_cells(p,h,grid)
  np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c);assert a.shape==b.shape==c.shape and a.dtype==b.dtype==c.dtype;checked+=2
# Saved actual physical poses are used solely to exercise sensor geometry, never policy inputs.
frames_saved=[json.loads(line) for line in (ROOT/'artifacts/extended_integrated_candidate_9167_30_static_20261011_attempt1/gppo/mission_frames.jsonl').read_text().splitlines()]
from utils.scenario_config import load_and_validate_scenario
from utils.obstacle_manager import ObstacleManager
cfg=load_and_validate_scenario(ROOT/'configs/scenarios/extended_connected_gppo_20261010.yaml');ob=ObstacleManager(cfg['environment']);ob.load_from_file(cfg['environment']['map_file']);ob.reset();g=ob.get_occupancy_grid();frame=ob.frame
old=IdealSensor(dict(fov=120.,range=10.,los_occlusion=True),frame=frame);new=module.IdealSensor(dict(fov=120.,range=10.,los_occlusion=True,visible_cells_cache_entries=128),frame=frame)
poses=[]
for row in frames_saved[:12]+frames_saved[-3:]:
 for value in row['physical_poses_diagnostic_only'].values():
  p=np.asarray(value['position']);h=value['heading'];a=old._visible_cells(p,h,g);b=new._visible_cells(p,h,g);np.testing.assert_array_equal(a,b);checked+=1
last=frames_saved[-1]['physical_poses_diagnostic_only'];poses=[(np.asarray(v['position']),v['heading']) for v in last.values()]
for p,h in poses:new._visible_cells(p,h,g)
times={}
for name,sensor in [('old',old),('cached',new)]:
 start=time.perf_counter()
 for repeat in range(10):
  for p,h in poses:sensor._visible_cells(p,h,g)
 times[name]=time.perf_counter()-start
result=dict(passed=True,exact_ordered_array_comparisons=checked,random_test_seed=62800,repeated_still_pose_calls=300,cpu_seconds=times,isolated_still_pose_sensor_speedup=times['old']/times['cached'],sensor_parameters_unchanged=True,physical_model_unchanged=True,truth_map_used_only_by_sensor_equality_test=True,production_installed=False,acceptance_claim=False,scope='Pure sensor output equality and isolated repeated still-pose CPU benchmark, not whole-runtime or task efficiency guarantee.')
(STAGE/'staged_equality_and_cpu.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
