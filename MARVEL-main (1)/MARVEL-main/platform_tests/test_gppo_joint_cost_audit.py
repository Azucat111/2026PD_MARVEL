from scripts.audit_gppo_joint_cost import two_task_costs

def test_competing_assignment_shows_real_greedy_gap():
 assert two_task_costs([[1,2],[1.1,100]],[[False,False],[False,False]])=={'optimal':3.1,'greedy':101.0}

def test_masks_and_single_uav_are_respected():
 assert two_task_costs([[1],[2]],[[False],[False]]) is None
 assert two_task_costs([[1,2],[1.1,100]],[[True,False],[False,True]])=={'optimal':3.1,'greedy':3.1}
