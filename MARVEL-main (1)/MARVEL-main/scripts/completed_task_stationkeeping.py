"""Independent extended-only lifecycle experiment; never changes production code."""
def stationkeep_completed_tasks(scheduler,actions,ordinary_controller=None):
    if scheduler._search_recovery is None or not scheduler.runtime.tasks.all_complete():return actions
    active=set(map(int,scheduler.search_assignments.values()))
    active.update(int(uid) for uid,_ in scheduler.relay_assignments.values())
    active.update(map(int,scheduler.safety.active_uav_ids))
    result=list(actions)
    for index,robot in enumerate(scheduler.runtime.robots):
        uid=int(robot.robot_id)
        if uid in active:continue
        result[index]=(robot.position.copy(),float(robot.heading))
        scheduler._search_recovery.reset_uid(uid)
        if ordinary_controller is not None and ordinary_controller is not scheduler._search_recovery:
            ordinary_controller.reset_uid(uid)
    return result
