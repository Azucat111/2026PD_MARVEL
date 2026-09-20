"""
Test MARVEL checkpoint with original environment to verify it works.
This uses the exact environment from MARVEL paper implementation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np
import torch
from parameter import *
from utils.env import Env
from utils.agent import Agent
from utils.node_manager import NodeManager
from utils.ground_truth_node_manager import GroundTruthNodeManager
from utils.model import PolicyNet


def test_original_marvel():
    """Run one episode with MARVEL original environment and checkpoint."""
    print(f"{'='*60}")
    print("MARVEL Original Environment Test")
    print(f"{'='*60}")
    print(f"Checkpoint: {load_path}/checkpoint.pth")
    print(f"Agents: {N_AGENTS}")
    print(f"Max steps: {MAX_EPISODE_STEP}")
    print(f"Sensor range: {SENSOR_RANGE}m")
    print()

    # Initialize environment (episode 0 will use first map in maps_medium)
    env = Env(episode_index=0, fov=FOV, sensor_range=SENSOR_RANGE, plot=False)
    print(f"Map size: {env.ground_truth_size}")
    print(f"Initial positions:\n{env.robot_locations}")
    print(f"Initial angles: {env.angles}")
    print()

    # Initialize device and policy network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
    policy_net.to(device)

    # Load checkpoint
    checkpoint_path = Path(ROOT) / load_path / "checkpoint.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    episode = checkpoint.get('episode', -1)
    print(f"Loading checkpoint from episode {episode}")

    policy_net.load_state_dict(checkpoint['policy_model'])
    policy_net.eval()
    print("Checkpoint loaded successfully\n")

    # Initialize node managers
    node_manager = NodeManager(FOV, SENSOR_RANGE, plot=False)
    ground_truth_node_manager = GroundTruthNodeManager(
        node_manager,
        env.ground_truth_info,
        SENSOR_RANGE,
        device=device,
        plot=False
    )

    # Initialize agents
    agents = []
    for i in range(N_AGENTS):
        agent = Agent(
            i,
            policy_net,
            FOV,
            env.angles[i],
            SENSOR_RANGE,
            node_manager,
            ground_truth_node_manager,
            device,
            plot=False
        )
        agents.append(agent)

    # Run episode
    step = 0

    # Initial graph update
    for robot in agents:
        robot.update_graph(env.belief_info, env.robot_locations[robot.id].copy())
    for robot in agents:
        robot.update_planning_state(env.robot_locations)

    while step < MAX_EPISODE_STEP:
        # Select actions
        actions = []
        for agent in agents:
            obs = agent.get_observation()
            next_position, _, _, heading_idx = agent.select_next_waypoint(obs, greedy=True)
            heading = float(heading_idx) * (360.0 / NUM_ANGLES_BIN)
            actions.append((next_position, heading))

        # Execute actions (simplified - just update positions)
        new_locations = []
        for i, (waypoint, heading) in enumerate(actions):
            # Move towards waypoint (simplified dynamics)
            direction = waypoint - env.robot_locations[i]
            dist = np.linalg.norm(direction)
            if dist > VELOCITY:
                direction = direction / dist * VELOCITY
            new_pos = env.robot_locations[i] + direction
            new_locations.append(new_pos)

            # Update belief
            robot_cell = ((new_pos - np.array([env.belief_info.map_origin_x,
                                               env.belief_info.map_origin_y]))
                         / CELL_SIZE).astype(int)
            env.update_robot_belief(robot_cell, heading)

        env.robot_locations = np.array(new_locations)

        # Update graphs for next step
        for robot in agents:
            robot.update_graph(env.belief_info, env.robot_locations[robot.id].copy())
        for robot in agents:
            robot.update_planning_state(env.robot_locations)

        # Calculate exploration rate
        explored_cells = np.sum(env.robot_belief != UNKNOWN)
        total_free_cells = np.sum(env.ground_truth == FREE)
        exploration_rate = explored_cells / total_free_cells if total_free_cells > 0 else 0

        step += 1

        if step % 10 == 0 or step == 1:
            print(f"Step {step:3d}: Exploration {exploration_rate:.2%}")

    # Final results
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}")
    print(f"Final exploration rate: {exploration_rate:.2%}")
    print(f"Total steps: {step}")
    print(f"Expected: >80% for working checkpoint")

    if exploration_rate > 0.5:
        print("\n✓ Checkpoint appears to be working correctly")
        return True
    else:
        print("\n✗ Checkpoint may have issues or environment mismatch")
        return False


if __name__ == "__main__":
    test_original_marvel()
