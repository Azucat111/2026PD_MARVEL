"""Verify handoff bytes, CPU models and focused contracts without legacy paths."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase2', action='store_true',
                        help='Also test atomic scheduling and verify intentional source changes against the handoff')
    args = parser.parse_args()
    manifest = json.loads((PROJECT/'handoff/cloud/snapshot_manifest.json').read_text())
    mismatches = [p for p, row in manifest['source'].items()
                  if hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() != row['sha256']]
    intentional = {
        'integrations/gppo/scheduler.py', 'integrations/gppo/mesh_relay.py',
        'utils/policy_adapter.py',
    } if args.phase2 else set()
    assert not set(mismatches) - intentional, mismatches
    for path in mismatches:
        repo_path = PROJECT.relative_to(PROJECT.parents[1]) / path
        original = subprocess.check_output(
            ['git', 'show', f'5692ca4fe5c56f398707deea87c1efc844297b71:{repo_path}'],
            cwd=PROJECT)
        assert hashlib.sha256(original).hexdigest() == manifest['source'][path]['sha256']
    subprocess.run([sys.executable, '-B', str(PROJECT/'handoff/cloud/smoke.py')],
                   cwd=PROJECT, check=True)
    models = json.loads((PROJECT/'handoff/cloud/checkpoint_manifest.json').read_text())
    row = next(r for r in models if r['name'] == 'Frozen native GPPO')
    checkpoint = PROJECT/row['cloud_path']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['cloud_sha256']
    import pytest
    results = dict(passed=0, failed=0, skipped=0)
    class PortableCheckpoint:
        def pytest_collection_modifyitems(self, items):
            for item in items:
                if Path(item.module.__file__).name == 'test_gppo_search_scenario.py':
                    item.module.FROZEN_CHECKPOINT = checkpoint
                if item.name.startswith('test_known_free_relay_exit_moves_with_missing_coarse_connector'):
                    directory = PROJECT/'artifacts/extended_gppo_joint_batch_20261012_attempt18'
                    if not (directory/'readonly_static64_112_control_inputs.json').exists():
                        item.add_marker(pytest.mark.skip(reason='historical public route fixture absent from handoff'))
        def pytest_runtest_logreport(self, report):
            if report.when == 'call' and report.passed:
                results['passed'] += 1
            elif report.failed:
                results['failed'] += 1
            elif report.skipped:
                results['skipped'] += 1
    files = ('test_joint_shadow.py', 'test_rule_scheduling_baseline.py',
             'test_joint_completion_metric.py', 'test_gppo_search_scenario.py',
             'test_extended_relay_arrival_priority.py', 'test_extended_search_catalog.py',
             'test_search_deadline_bound.py', 'test_observed_search_reachability.py',
             'test_search_route_support.py')
    if args.phase2:
        files += ('test_joint_atomic.py', 'test_extended_relay_augmentation.py',
                  'test_observed_mesh_relay.py', 'test_extended_relay_budget.py',
                  'test_known_relay_route.py', 'test_observed_relay_probe.py')
    code = pytest.main(['-p', 'no:cacheprovider', '-q'] +
                       [str(PROJECT/'platform_tests'/p) for p in files], plugins=[PortableCheckpoint()])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(source_files_checked=len(manifest['source']),
        intentional_source_changes=mismatches, source_mismatches=[], model_smoke='8 SHA256 + strict CPU load + masked forward',
        portable_test_checkpoint=str(checkpoint.relative_to(PROJECT)),
        portability_change='test module path variable only; no production/test file rewritten',
        tests=results, exit_code=int(code)), indent=2))
    raise SystemExit(code)

if __name__ == '__main__':
    main()
