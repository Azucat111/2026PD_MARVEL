"""Read-only protocol/metric audit for final technical acceptance planning."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.checkpoint import load_frozen_gppo
from utils.scenario_config import load_and_validate_scenario
from utils.obstacle_manager import ObstacleManager
_,checkpoint=load_frozen_gppo("/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth","cpu")
mixed=checkpoint["mixed_config"]
print(json.dumps({k:mixed.get(k) for k in ["comm_range","comm_max_hops","comm_delay_ms","control_step_ms",
    "max_episode_steps","max_search_uavs","max_relay_uavs","activation_threshold",
    "connectivity_threshold","relay_release_threshold"]},indent=2))
for scenario in ("extended_live_gppo.yaml","extended_live_gppo_full.yaml","baseline_maps_test.yaml"):
    config=load_and_validate_scenario(ROOT/"configs/scenarios"/scenario)
    ob=ObstacleManager(config["environment"]);ob.load_from_file(config["environment"]["map_file"])
    grid=ob.get_occupancy_grid()
    print(json.dumps({"scenario":scenario,"extent":ob.frame.describe(),"free_cells":int((grid==0).sum()),
        "all_cells":grid.size,"free_fraction":float((grid==0).mean()),
        "dynamic_obstacles":config.get("dynamic_obstacles",[])},indent=2))
