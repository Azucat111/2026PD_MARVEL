"""Run existing regression baseline in memory from HEAD, or current working tree.
Single-thread CPU libraries avoid oversubscription; no source files are reverted.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from pathlib import Path
import sys, subprocess, importlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
if "--baseline" in sys.argv:
    for name, rel in (
        ("utils.simulation_runtime", "utils/simulation_runtime.py"),
        ("integrations.gppo.scheduler", "integrations/gppo/scheduler.py"),
        ("utils.policy_adapter", "utils/policy_adapter.py"),
    ):
        module = importlib.import_module(name)
        full = subprocess.check_output(
            ["git", "ls-files", "--full-name", rel], cwd=ROOT, text=True).strip()
        assert full, rel
        source = subprocess.check_output(["git", "show", "HEAD:" + full], cwd=ROOT, text=True)
        exec(compile(source, str(ROOT / rel), "exec"), module.__dict__)
    arguments = ["platform_tests", "--ignore=platform_tests/test_extended_search_recovery.py"]
elif "--focused" in sys.argv:
    arguments = [str(p.relative_to(ROOT)) for p in sorted((ROOT/"platform_tests").glob("test_*.py"))
        if p.name.startswith("test_marvel_native") or p.name in {
            "test_extended_search_recovery.py", "test_extended_search_service_scan.py", "test_platform.py",
            "test_gppo_search_scenario.py", "test_gppo_runtime_state_parity.py",
            "test_observed_search_reachability.py","test_search_deadline_bound.py","test_search_route_support.py",
            "test_observed_belief.py","test_extended_observed_travel.py","test_extended_node_lattice.py",
            "test_extended_motion_shield.py","test_exploration_coverage_basis.py",
            "test_rule_scheduling_baseline.py","test_search_efficiency.py",
            "test_scheduler_candidate_filter.py"}]
else:
    arguments = ["platform_tests"]
import pytest
raise SystemExit(pytest.main(arguments + ["-q", "-p", "no:cacheprovider"]))
