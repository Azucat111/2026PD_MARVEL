"""Read-only geometric relay bound at saved initial fleet poses."""
from pathlib import Path
import argparse,json,numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.gppo.event_sources import ObservedRelayDemandBuilder
from utils.communication_model import CommunicationModel
p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();folder=Path(a.folder)
config=json.loads((folder/'config.json').read_text());initial=json.loads((folder/'initial.json').read_text())
base=np.asarray(config['task_scheduler']['base_position']);positions=np.asarray(initial['initial_positions'])
profile=config['_gppo_profile'];r=float(profile['comm_range']);hops=int(profile['comm_max_hops'])
if hops!=2:raise ValueError('This diagnostic requires the frozen two-hop profile')
# GPPO profile's frozen two-hop rule: arbitrary helpers cannot connect a fixed target farther than2R.
dist=np.linalg.norm(positions-base,axis=1);inside=np.flatnonzero(dist<=2*r+1e-9)
snapshot=ObservedRelayDemandBuilder(comm_range=r,max_hops=hops).snapshot(dict(enumerate(positions)),base_position=base)
comm=CommunicationModel({'comm_range':r})
result={'largest_peer_component_fraction':comm.connectivity_ratio(comm.get_topology(positions)),
'base_two_hop_fraction':snapshot.connectivity_ratio,
'fixed_pose_geometry_only':True,'not_a_future_motion_impossibility_claim':True,
'base':base.tolist(),'comm_range':r,'max_hops':2,'uids_outside_any_two_hop_bridge':np.flatnonzero(dist>2*r+1e-9).tolist(),
'fixed_pose_connectivity_upper_bound':len(inside)/len(positions),
'with_two_helpers_relocated_others_fixed_upper_bound':min(1.,(len(inside)+2)/len(positions)),
'distances_m':dist.tolist()}
(folder/'relay_geometry_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
