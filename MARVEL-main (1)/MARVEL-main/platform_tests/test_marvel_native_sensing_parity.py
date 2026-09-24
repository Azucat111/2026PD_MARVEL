"""MARVEL-native sensing / exploration-rate parity tests.

The native mode must reproduce the frozen MARVEL belief pipeline exactly:

    sensor_work_heading -> fov_sweep -> collision_check -> robot_belief
    explored_rate = sum(belief == 255) / sum(ground_truth == 255)

The frozen side runs in a subprocess, because both repos ship a top-level
``utils`` package.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.marvel_maps import load_marvel_native_map
from utils.marvel_sensing import (
    FREE,
    MAX_COLLISION,
    OCCUPIED,
    UNKNOWN,
    MarvelNativeBelief,
    calculate_fov_boundaries,
    fov_sweep,
)
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime

FROZEN_REPO = Path("/home/nick/MARVEL")
FROZEN_CHECKPOINT = (
    FROZEN_REPO
    / "artifacts"
    / "phase14_v9_frozen"
    / "gppo_phase14_v9_ckpt80.pth"
)

NATIVE_SCENARIO = ROOT / "configs" / "scenarios" / "marvel_native_test.yaml"

MAP_NAME = "1.png"
CELL_SIZE = 0.4
SENSOR_RANGE = 10.0
FOV = 120.0

requires_frozen = pytest.mark.skipif(
    not FROZEN_REPO.exists(),
    reason=f"frozen MARVEL repo not available at {FROZEN_REPO}",
)


# ======================================================================
# Frozen reference probe
# ======================================================================

FROZEN_PROBE = textwrap.dedent(
    '''
    import sys, json
    sys.path.insert(0, "/home/nick/MARVEL")
    import numpy as np
    from skimage import io
    from skimage.measure import block_reduce
    from utils.sensor import sensor_work_heading

    image = io.imread("maps_test/1.png", 1).astype(int)
    gt = block_reduce(image, 2, np.min)
    gt = (gt > 150) | ((gt <= 80) & (gt >= 50))
    gt = (gt * 254 + 1).astype(np.int32)

    CELL = 0.4
    RANGE_CELLS = round(10.0 / CELL)

    cases = json.loads(sys.argv[1])
    out = {}
    for name, sequence in cases.items():
        belief = np.ones(gt.shape, dtype=np.int32) * 127
        for cell, heading in sequence:
            belief = sensor_work_heading(
                np.array([int(cell[0]), int(cell[1])]), RANGE_CELLS,
                belief, gt, float(heading), 120.0,
            )
        out[name] = {
            "free": int((belief == 255).sum()),
            "occupied": int((belief == 1).sum()),
            "unknown": int((belief == 127).sum()),
            "rate": float((belief == 255).sum() / (gt == 255).sum()),
            "flat": belief.astype(np.int64).flatten().tolist(),
        }
    print(json.dumps(out))
    '''
)

# Activation probe: run the REAL frozen maybe_activate() gate against an
# explicit explored_rate sequence.  Only the two collaborators that are
# irrelevant to the activation decision are stubbed.
FROZEN_ACTIVATION_PROBE = textwrap.dedent(
    '''
    import sys, json
    sys.path.insert(0, "/home/nick/MARVEL")
    import numpy as np
    from utils.scenario_env import ScenarioEnv
    from utils.explore_search_coordinator import ExploreSearchCoordinator
    from utils.task_graph import TaskType

    env = ScenarioEnv(episode_index=0, fov=120.0, n_agents=4,
                      sensor_range=10.0, map_dir="maps_test",
                      scenario_seed=12345)
    coord = ExploreSearchCoordinator(
        env=env, robots=[None] * 4, task_manager=None, gppo_model=None,
        device="cpu", max_search_uavs=2, survivor_count=4, seed=12345,
    )

    class _Task:
        completed = False
        active = True

    class _TaskManager:
        """Only records subtask registration; the activation gate is what
        this probe measures."""
        def __init__(self):
            self.calls = 0
        def add_subtask(self, *args, **kwargs):
            self.calls += 1
            return self.calls
        def get_subtask(self, subtask_id):
            return _Task()

    coord.task_manager = _TaskManager()
    coord._allocate_search_subtasks = lambda **kwargs: None

    rates = json.loads(sys.argv[1])
    first_activation = None
    decisions = []
    for index, rate in enumerate(rates):
        env.explored_rate = float(rate)
        activated = bool(coord.maybe_activate(index))
        decisions.append(activated)
        if activated and first_activation is None:
            first_activation = index

    print(json.dumps({
        "threshold": float(coord.activation_threshold),
        "decisions": decisions,
        "first_activation": first_activation,
        "search_activated": bool(coord.search_activated),
    }))
    '''
)


@pytest.fixture(scope="module")
def frozen_sensing(tmp_path_factory):
    if not FROZEN_REPO.exists():
        pytest.skip("frozen MARVEL repo not available")

    script = tmp_path_factory.mktemp("frozen_sense") / "probe.py"
    script.write_text(FROZEN_PROBE, encoding="utf-8")

    sequences = {
        "A_one_update": [[[40, 40], 270.0]],
        "B_five_updates": [
            [[40, 40], 270.0],
            [[42, 41], 258.0],
            [[44, 42], 246.0],
            [[46, 44], 234.0],
            [[48, 46], 222.0],
        ],
        "C_movement": [
            [[125, 125], 0.0],
            [[126, 125], 15.0],
            [[127, 126], 30.0],
            [[128, 127], 45.0],
            [[129, 129], 60.0],
            [[130, 130], 90.0],
        ],
        "D_fov_boundary_wrap": [[[60, 60], 350.0]],
        "E_obstacle_edge": [[[80, 120], 180.0]],
        "F_own_cell_free": [[[100, 100], 0.0]],
    }

    result = subprocess.run(
        [sys.executable, str(script), json.dumps(sequences)],
        capture_output=True,
        text=True,
        cwd=str(FROZEN_REPO),
    )

    if result.returncode != 0:
        pytest.skip(f"frozen sensing probe failed: {result.stderr[-400:]}")

    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def native_ground_truth():
    _occupancy, _origin, ground_truth = load_marvel_native_map(
        ROOT / "maps_test" / MAP_NAME
    )

    return ground_truth


def _belief():
    _occupancy, _origin, ground_truth = load_marvel_native_map(
        ROOT / "maps_test" / MAP_NAME
    )

    return MarvelNativeBelief(
        ground_truth=ground_truth,
        cell_size=CELL_SIZE,
        sensor_range=SENSOR_RANGE,
        fov=FOV,
    )


# ======================================================================
# 3. Exact parity tests
# ======================================================================

@requires_frozen
@pytest.mark.parametrize(
    "case",
    [
        "A_one_update",
        "B_five_updates",
        "C_movement",
        "D_fov_boundary_wrap",
        "E_obstacle_edge",
        "F_own_cell_free",
    ],
)
def test_native_belief_matches_frozen(frozen_sensing, case):
    """Identical belief mask, explored free count and explored_rate."""

    reference = frozen_sensing[case]

    sequences = {
        "A_one_update": [[[40, 40], 270.0]],
        "B_five_updates": [
            [[40, 40], 270.0],
            [[42, 41], 258.0],
            [[44, 42], 246.0],
            [[46, 44], 234.0],
            [[48, 46], 222.0],
        ],
        "C_movement": [
            [[125, 125], 0.0],
            [[126, 125], 15.0],
            [[127, 126], 30.0],
            [[128, 127], 45.0],
            [[129, 129], 60.0],
            [[130, 130], 90.0],
        ],
        "D_fov_boundary_wrap": [[[60, 60], 350.0]],
        "E_obstacle_edge": [[[80, 120], 180.0]],
        "F_own_cell_free": [[[100, 100], 0.0]],
    }

    belief = _belief()

    for cell, heading in sequences[case]:
        belief.observe(cell, heading)

    # Belief mask identical, cell for cell.
    assert belief.belief.astype(np.int64).flatten().tolist() == (
        reference["flat"]
    )

    # Explored free-cell count identical.
    assert belief.explored_free_count == reference["free"]

    # Occupied / unknown counts identical.
    assert int((belief.belief == OCCUPIED).sum()) == (
        reference["occupied"]
    )
    assert int((belief.belief == UNKNOWN).sum()) == (
        reference["unknown"]
    )

    # explored_rate identical.
    assert belief.explored_rate == pytest.approx(
        reference["rate"], abs=1e-15
    )


@requires_frozen
def test_five_updates_accumulate_identically(frozen_sensing):
    """Belief state is shared and monotone across updates."""

    belief = _belief()

    sequence = [
        [[40, 40], 270.0],
        [[42, 41], 258.0],
        [[44, 42], 246.0],
        [[46, 44], 234.0],
        [[48, 46], 222.0],
    ]

    counts = []

    for cell, heading in sequence:
        belief.observe(cell, heading)
        counts.append(belief.explored_free_count)

    # Once a cell is observed free it stays free; coverage never decreases.
    assert counts == sorted(counts)
    assert counts[-1] == frozen_sensing["B_five_updates"]["free"]


def test_belief_requires_no_import_of_frozen_repo():
    """The port is self-contained; only the tests shell out to frozen."""

    belief = _belief()

    assert belief.belief.shape == (250, 250)
    assert belief.free_cell_total == 22402
    assert belief.sensor_range_cells == 25


# ======================================================================
# 5. Own-cell handling
# ======================================================================

@requires_frozen
def test_frozen_writes_the_robot_own_cell(native_ground_truth):
    """Frozen collision_check writes (x0, y0) when that cell is free."""

    belief = _belief()

    # A free cell with a clear view.
    free_cells = np.argwhere(native_ground_truth == FREE)
    cell = free_cells[len(free_cells) // 2]

    assert native_ground_truth[cell[0], cell[1]] == FREE

    belief.observe((int(cell[1]), int(cell[0])), 0.0)

    # The robot's own cell is FREE in the belief map.
    assert belief.belief[cell[0], cell[1]] == FREE


@requires_frozen
def test_native_own_cell_free_matches_frozen(frozen_sensing):
    """Own-cell inclusion is exactly what frozen produces."""

    reference = frozen_sensing["F_own_cell_free"]

    belief = _belief()
    belief.observe([100, 100], 0.0)

    # Cell (100, 100) happens to be occupied on this map, and both
    # implementations agree on that too.
    assert belief.belief[100, 100] == OCCUPIED
    assert belief.belief.astype(np.int64).flatten().tolist() == (
        reference["flat"]
    )

    # Repeat from a cell that is genuinely free: the own cell is written.
    free_cells = np.argwhere(belief.ground_truth == FREE)
    row, col = free_cells[len(free_cells) // 2]

    belief.observe([int(col), int(row)], 0.0)

    assert belief.belief[row, col] == FREE


def test_extended_sensor_behaviour_is_untouched():
    """The extended IdealSensor still skips the robot's own cell."""

    from utils.sensor_models import IdealSensor

    sensor = IdealSensor({"fov": 120.0, "range": 10.0})

    grid = np.zeros((20, 20), dtype=np.uint8)

    cells = sensor._visible_cells(
        np.asarray((10.0, 10.0)), 0.0, grid
    )

    assert (10, 10) not in {tuple(c) for c in cells}


# ======================================================================
# collision_check semantics
# ======================================================================

def _ray(x0, y0, x1, y1, ground_truth, belief):
    """Call ``collision_check`` with the frozen caller's argument types.

    The frozen Bresenham loop seeds ``x``/``y`` from ``x0``/``y0`` and
    relies on them staying integer, while the ray endpoint arrives as a
    float from ``x0 + cos(angle) * range``.  Passing Python scalars raises
    inside the frozen code, so the tests use the real convention.
    """

    from utils.marvel_sensing import collision_check

    return collision_check(
        np.int64(x0),
        np.int64(y0),
        np.float64(x1),
        np.float64(y1),
        ground_truth,
        belief,
    )


def test_ray_stops_after_max_collision_occupied_cells():
    """At most MAX_COLLISION occupied cells are consumed per ray."""

    assert MAX_COLLISION == 2

    ground_truth = np.full((20, 20), FREE, dtype=np.int32)
    ground_truth[:, 10:] = OCCUPIED

    belief = np.ones((20, 20), dtype=np.int32) * UNKNOWN

    # Ray straight into the wall.
    belief = _ray(2, 5, 18, 5, ground_truth, belief)

    # Free cells up to the wall are written.
    assert belief[5, 2] == FREE
    assert belief[5, 9] == FREE
    # The first occupied cell is written as occupied, then the ray ends.
    assert belief[5, 10] == OCCUPIED
    # Beyond it stays unknown.
    assert belief[5, 11] == UNKNOWN
    assert belief[5, 18] == UNKNOWN


def test_ray_endpoint_is_never_written():
    ground_truth = np.full((20, 20), FREE, dtype=np.int32)
    belief = np.ones((20, 20), dtype=np.int32) * UNKNOWN

    belief = _ray(2, 5, 12, 5, ground_truth, belief)

    assert belief[5, 2] == FREE
    assert belief[5, 11] == FREE
    # The endpoint cell itself is skipped by the frozen guard.
    assert belief[5, 12] == UNKNOWN


def test_ray_terminates_out_of_bounds():
    ground_truth = np.full((10, 10), FREE, dtype=np.int32)
    belief = np.ones((10, 10), dtype=np.int32) * UNKNOWN

    belief = _ray(2, 5, 100, 5, ground_truth, belief)

    assert belief[5, 2] == FREE
    assert belief[5, 9] == FREE


def test_fov_boundaries_wrap_correctly():
    start, end = calculate_fov_boundaries(0.0, 120.0)

    assert start == 300.0
    assert end == 60.0

    angles = np.degrees(fov_sweep(start, end, 0.5))

    assert len(angles) > 0

    # The sweep wraps through 0: it covers [300, 360) and [0, 60], and
    # nothing in the blind arc (60, 300).
    wrapped = angles[angles >= 300.0]
    direct = angles[angles <= 60.0]

    assert len(wrapped) > 0
    assert len(direct) > 0
    assert wrapped.min() >= 300.0
    assert wrapped.max() < 360.0
    assert direct.max() <= 60.0 + 1e-9

    blind = angles[(angles > 60.0 + 1e-9) & (angles < 300.0 - 1e-9)]

    assert len(blind) == 0


# ======================================================================
# 4. Search activation parity at 0.30
# ======================================================================

class _FixedRateBelief:
    """Stand-in belief whose explored_rate is an exact requested value.

    Integer cell counts cannot hit exactly 0.30 on this map
    (0.30 * 22402 = 6720.6), so the threshold operator is probed with exact
    floats on both sides.
    """

    def __init__(self, rate: float):
        self._rate = float(rate)
        self.free_total = 22402

    @property
    def explored_rate(self) -> float:
        return self._rate


def _native_gppo_scheduler():
    from integrations.gppo.checkpoint import load_frozen_gppo
    from integrations.gppo.config_profile import prepare_gppo_config
    from integrations.gppo.scheduler import GPPOTaskScheduler
    from integrations.gppo.search_scenario import (
        initialize_search_scenario,
    )

    _, checkpoint = load_frozen_gppo(str(FROZEN_CHECKPOINT), device="cpu")

    config = load_and_validate_scenario(NATIVE_SCENARIO)
    config["scenario"]["random_seed"] = 7
    config["task_scheduler"] = {
        "mode": "gppo",
        "checkpoint": str(FROZEN_CHECKPOINT),
        "base_position": [0.0, 0.0],
        "search_scenario_seed": 7,
    }
    config = prepare_gppo_config(config, checkpoint)

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    scheduler = GPPOTaskScheduler(
        runtime,
        checkpoint_path=str(FROZEN_CHECKPOINT),
        device="cpu",
        scheduler_config=config["task_scheduler"],
    )

    truth = initialize_search_scenario(runtime)
    scheduler.install_search_scenario(truth.public_heat_points())

    runtime.tasks.tasks["T2_target_search"].status = "active"

    return runtime, scheduler, config


@requires_frozen
def test_search_activation_matches_frozen_on_identical_rates(tmp_path):
    """Both must make the same decision on every high-level update."""

    from utils.marvel_sensing import FREE, UNKNOWN

    runtime, scheduler, config = _native_gppo_scheduler()

    assert config["_gppo_profile"]["activation_threshold"] == (
        pytest.approx(0.30)
    )

    free = runtime.native_belief.free_mask()
    rows, cols = np.nonzero(free)
    total = len(rows)

    # Coverage sweep expressed as integer cell counts, so both sides see
    # the *identical* rate at every step.
    fractions = [0.10, 0.20, 0.28, 0.29, 0.295, 0.2999, 0.30, 0.31, 0.50]
    takes = [int(fraction * total) for fraction in fractions]
    rates = [take / total for take in takes]

    script = tmp_path / "activation_probe.py"
    script.write_text(FROZEN_ACTIVATION_PROBE, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), json.dumps(rates)],
        capture_output=True,
        text=True,
        cwd=str(FROZEN_REPO),
    )

    if result.returncode != 0:
        pytest.skip(f"frozen activation probe failed: {result.stderr[-400:]}")

    frozen = json.loads(result.stdout.strip().splitlines()[-1])

    assert frozen["threshold"] == pytest.approx(0.30)

    decisions = []
    first_activation = None

    for index, take in enumerate(takes):
        runtime.native_belief.belief[:] = UNKNOWN
        runtime.native_belief.belief[rows[:take], cols[:take]] = FREE

        assert runtime.exploration_rate == pytest.approx(rates[index])

        # Activation is one-shot on both sides, so the latch is NOT reset.
        activated = bool(scheduler._maybe_activate_search())
        decisions.append(activated)

        if activated and first_activation is None:
            first_activation = index

    # Identical decision at every high-level update, including the latch.
    assert decisions == frozen["decisions"]
    assert first_activation == frozen["first_activation"]

    # The crossing point is where the rate actually reaches 0.30.
    expected_first = next(
        index
        for index, rate in enumerate(rates)
        if rate >= 0.30
    )

    assert first_activation == expected_first
    assert decisions[:expected_first] == [False] * expected_first

    # Exactly one activation, then latched.
    assert sum(decisions) == 1


@requires_frozen
def test_search_activation_operator_is_ge_at_exact_threshold(tmp_path):
    """Inclusive >= on both sides, probed with exact floats."""

    runtime, scheduler, _config = _native_gppo_scheduler()

    exact_rates = [0.2999999, 0.30, 0.3000001]

    script = tmp_path / "exact_probe.py"
    script.write_text(FROZEN_ACTIVATION_PROBE, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), json.dumps(exact_rates)],
        capture_output=True,
        text=True,
        cwd=str(FROZEN_REPO),
    )

    if result.returncode != 0:
        pytest.skip(f"frozen activation probe failed: {result.stderr[-400:]}")

    frozen = json.loads(result.stdout.strip().splitlines()[-1])

    # Frozen: inactive strictly below, active at exactly 0.30.
    assert frozen["decisions"] == [False, True, False]
    assert frozen["first_activation"] == 1

    runtime.native_belief = _FixedRateBelief(0.2999999)

    assert runtime.exploration_rate == pytest.approx(0.2999999)
    assert not scheduler._maybe_activate_search()

    runtime.native_belief = _FixedRateBelief(0.30)

    assert runtime.exploration_rate == 0.30
    assert scheduler._maybe_activate_search()

    # Latch: a second call on the same scheduler returns False.
    assert not scheduler._maybe_activate_search()
