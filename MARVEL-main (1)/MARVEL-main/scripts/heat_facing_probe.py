"""Throwaway extended KinematicSimple heading experiment; no production import."""
import numpy as np
def heat_facing_action(robot,anchor,command,recovery,reach_radius):
 if recovery is None:return command
 position=np.asarray(robot.position,dtype=float);anchor=np.asarray(anchor,dtype=float)
 if (int(robot.robot_id) in recovery.sensing_uids
     or int(robot.robot_id) in recovery.prefix_headings
     or np.linalg.norm(anchor-position)<=reach_radius+1e-9
     or np.linalg.norm(np.asarray(command[0])-position)<=.1
     or not recovery.segment_safe(position,anchor)):return command
 direction=anchor-position
 return command[0],float(np.degrees(np.arctan2(direction[1],direction[0]))%360.)
