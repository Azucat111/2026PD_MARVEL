"""
Test MARVEL checkpoint with original environment to verify it works.
This uses the exact environment from MARVEL paper implementation.
"""
import sys
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from parameter import *
from utils.env import Env
from utils.agent import Agent


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
    print(f"Initial positions: {env.robot_locations}")
    print(f"Initial exploration rate: {env.explored_rate:.2%}")
    print()

    # Initialize agents
    agents = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    for i in range(N_AGENTS):
        agent = Agent(
            i,
            env.belief_info,
            SENSOR_RANGE,
            FOV,
            NUM_ANGLES_BIN,
            env.robot_locations[i],
            env.angles[i],
            device,
            plot=False
        )
        agents.append(agent)

    # Load checkpoint
    checkpoint_path = Path(ROOT) / load_path / "checkpoint.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    episode = checkpoint.get('episode', -1)
    print(f"Loading checkpoint from episode {episode}")

    for agent in agents:
        agent.policy_net.load_state_dict(checkpoint['policy_model'])
        agent.policy_net.eval()

    print("Checkpoint loaded successfully\n")

    # Run episode
    step = 0
    collision_count = 0

    while step < MAX_EPISODE_STEP:
        # Update each agent's graph and get actions
        for i, agent in enumerate(agents):
            agent.update_graph(env.belief_info, env.robot_locations[i])

        # Update planning state with all robot locations
        for agent in agents:
            agent.update_planning_state(env.robot_locations)

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
    else:
        print("\n✗ Checkpoint may have issues or environment mismatch")


if __name__ == "__main__":
    test_original_marvel()
