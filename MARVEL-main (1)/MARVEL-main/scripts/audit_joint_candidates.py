"""Bounded CPU public-state sampling/replay; no training or H300 acceptance."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'scripts'), str(PROJECT/'handoff/cloud')]

from integrations.gppo.joint_shadow import (enumerate_joint, pack, unpack, snapshot_bytes,
                                            summarize, attach_actual)
from integrations.gppo.joint_shadow_recorder import JointShadowRecorder
from gppo_benchmark_domains import load_domain
from paired_eval import verified_asset
import benchmark_fair_search as benchmark


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_passivity(enabled, disabled):
    """Only wall-clock event timestamp is excluded; every semantic field stays."""
    checks = {}
    for name in ('initial.json', 'config.json', 'decisions.json', 'collisions.json', 'progress.csv'):
        assert (enabled/name).read_bytes() == (disabled/name).read_bytes(), name
        checks[name] = dict(exact_bytes_equal=True, sha256=sha(enabled/name))
    events = []
    for folder in (enabled, disabled):
        rows = json.loads((folder/'events.json').read_text())
        events.append([{k: v for k, v in row.items() if k != 'timestamp'} for row in rows])
    assert events[0] == events[1], 'semantic event trace differs'
    checks['events.json'] = dict(semantic_equal=True, excluded_fields=['top_level_timestamp'],
        events=len(events[0]), enabled_raw_sha256=sha(enabled/'events.json'),
        disabled_raw_sha256=sha(disabled/'events.json'))
    return dict(checks=checks, raw_events_byte_equal=False,
                scope='all recorded choices/events/progress, not an unrecorded full-pose trace')


def rebuild(source, out):
    """Reuse captured public inputs; no simulator/model runs or new samples."""
    capture_manifest = json.loads((source/'manifest.json').read_text())
    summaries = []
    all_events = []
    for case in capture_manifest['cases']:
        name = case['case']; original = source/name; folder = out/name
        folder.mkdir()
        captured = json.loads((original/'events.json').read_text())
        events = []
        for i, old in enumerate(captured):
            state_file = original/f'public_state_{i:03d}.json.gz'
            with gzip.open(state_file, 'rt') as f:
                state = unpack(json.load(f))
            event = enumerate_joint(state)
            assert event == enumerate_joint(state), 'rebuild not repeatable'
            attach_actual(event, old['actual_graphs'], [])
            for field in ('actual_assignments', 'actual_scheduler_commands',
                          'executor_ownership_after', 'purity_verified'):
                event[field] = old[field]
            event['repeat_verified'] = True
            event['capture_prototype_commit'] = 'ad864dd'
            shutil.copyfile(state_file, folder/state_file.name)
            events.append(event)
        (folder/'events.json').write_text(json.dumps(events, indent=2, allow_nan=False))
        summary = dict(summarize(events), case=name,
            sampled_physical_seconds=case['elapsed_seconds'],
            requested_control_steps=6, observed_control_boundaries=len(events),
            physical_window_complete_to_requested=case['mission_steps'] == 6,
            smoke_only=True, model_tensors_unchanged=case['model_tensors_unchanged'])
        summaries.append(summary);all_events.extend(events)
        (folder/'summary.json').write_text(json.dumps(summary, indent=2))
        raw = folder/'original_run'; raw.mkdir()
        for path in (original/'gppo').iterdir():
            if path.suffix in ('.json', '.csv'):
                # Preserve every original byte, including timestamps and oracle
                # scoring summaries; these are never fed back to enumerate_joint.
                with gzip.GzipFile(filename=str(raw/(path.name+'.gz')), mode='wb', mtime=0) as f:
                    f.write(path.read_bytes())
    passivity = verify_passivity(source/capture_manifest['cases'][0]['case']/'gppo',
                                 source/'passive_disabled/gppo')
    (out/'passivity.json').write_text(json.dumps(passivity, indent=2))
    (out/'capture_manifest.json').write_text(json.dumps(capture_manifest, indent=2))
    files = ('integrations/gppo/joint_shadow.py', 'integrations/gppo/joint_shadow_recorder.py',
             'scripts/audit_joint_candidates.py', 'platform_tests/test_joint_shadow.py')
    manifest = dict(cases=summaries, aggregate=summarize(all_events),
        capture_prototype_commit='ad864dd', base_commit=capture_manifest['base_head'],
        method='exact recorded public-state re-enumeration; no new simulation',
        enumeration_source_sha256={p: sha(PROJECT/p) for p in files},
        evidence_sha256={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('PUBLIC_STATE_REBUILD_OK', json.dumps(manifest['aggregate']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--wall-seconds-per-case', type=float, default=180)
    p.add_argument('--replay', type=Path)
    p.add_argument('--rebuild-from', type=Path)
    p.add_argument('--verify-passive', action='store_true')
    args = p.parse_args()
    if not 2 <= args.steps <= 20:
        p.error('Discovery smoke is bounded to 2..20 control steps')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.rebuild_from:
        rebuild(args.rebuild_from, args.out)
        return
    if args.replay:
        events = json.loads((args.replay/'events.json').read_text())
        for i, event in enumerate(events):
            with gzip.open(args.replay/f'public_state_{i:03d}.json.gz', 'rt') as f:
                state = unpack(json.load(f))
            reproduced = enumerate_joint(state)
            reproduced['purity_verified'] = True
            reproduced['repeat_verified'] = True
            attach_actual(reproduced, event['actual_graphs'], [])
            for key in ('candidates', 'dormant_catalog', 'potential_graphs', 'metrics', 'public_state_sha256'):
                assert reproduced[key] == event[key], (i, key)
        (args.out/'replay.json').write_text(json.dumps(dict(events=len(events), exact_replay=True), indent=2))
        print('EXACT_PUBLIC_REPLAY_OK', len(events))
        return

    manifest = json.loads((PROJECT/'handoff/cloud/checkpoint_manifest.json').read_text())
    base, base_sha = verified_asset(manifest, 'Frozen native GPPO')
    candidate, candidate_sha = verified_asset(manifest, 'November 11 guarded candidate')
    _, marvel_sha = verified_asset(manifest, 'MARVEL low-level actor')
    benchmark.CHECKPOINT = base
    scenario = PROJECT/'configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml'
    loader, runtime_type, adapter_type = (benchmark.load_and_validate_scenario,
                                         benchmark.SimulationRuntime, benchmark.MARVELPolicyAdapter)
    cases = [(30, False, 1280021), (30, True, 1280022),
             (60, False, 9164), (60, True, 1981204)]
    reports = []
    for uavs, dynamic, seed in cases:
        name = f'{uavs}_{"dynamic" if dynamic else "static"}_{seed}'
        folder = args.out/name
        folder.mkdir()
        recorders, models = [], []
        def model_digest(adapter):
            h = hashlib.sha256()
            for model in (adapter.policy_net, adapter.scheduler.allocator.adapter.model):
                for key, value in model.state_dict().items():
                    h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        class Runtime(runtime_type):
            def __init__(self, config):
                config['task_scheduler']['extended_checkpoint'] = str(candidate)
                config['task_scheduler']['extended_checkpoint_sha256'] = candidate_sha
                config['policy']['device'] = 'cpu'
                super().__init__(config)
        class Adapter(adapter_type):
            def setup(self):
                with benchmark.torch.random.fork_rng():
                    result = super().setup()
                recorder = JointShadowRecorder(self.scheduler, scenario=name)
                recorders.append(recorder)
                models.append((self, model_digest(self)))
                return result
        benchmark.SimulationRuntime, benchmark.MARVELPolicyAdapter = Runtime, Adapter
        benchmark.load_and_validate_scenario = lambda file: load_domain(loader, file, scenario, dynamic, .2)
        result = benchmark.run(folder, 'gppo', uavs, seed, args.steps,
                               physical_coverage=True, stop_on_completion=False,
                               wall_budget_seconds=args.wall_seconds_per_case)
        recorder = recorders[0]
        recorder.close()
        adapter, before_model = models[0]
        assert before_model == model_digest(adapter), 'Shadow/run changed frozen model tensors'
        assert not result['policy_execution_failures']
        events = recorder.events
        for i, state in enumerate(recorder.states):
            with gzip.GzipFile(filename=str(folder/f'public_state_{i:03d}.json.gz'), mode='wb', mtime=0) as f:
                f.write(snapshot_bytes(state))
        (folder/'events.json').write_text(json.dumps(events, indent=2, allow_nan=False))
        summary = summarize(events)
        summary.update(case=name, elapsed_seconds=result['elapsed_simulation_seconds'],
                       mission_steps=result['mission_steps'], smoke_only=True,
                       model_tensors_unchanged=True)
        (folder/'summary.json').write_text(json.dumps(summary, indent=2))
        reports.append(summary)
        print('SHADOW_CASE', json.dumps(summary), flush=True)
    benchmark.SimulationRuntime, benchmark.MARVELPolicyAdapter = runtime_type, adapter_type
    benchmark.load_and_validate_scenario = loader
    files = ['integrations/gppo/joint_shadow.py', 'integrations/gppo/joint_shadow_recorder.py',
             'scripts/audit_joint_candidates.py', 'integrations/gppo/scheduler.py',
             'integrations/gppo/event_allocator.py', 'integrations/gppo/task_graph.py',
             'integrations/gppo/mesh_relay.py', 'integrations/gppo/event_sources.py',
             'integrations/gppo/extended_travel.py', 'integrations/gppo/search_recovery.py',
             'integrations/gppo/route_support.py', 'integrations/gppo/observed_reachability.py',
             'integrations/gppo/search_deadline.py', 'utils/simulation_runtime.py',
             'utils/policy_adapter.py', 'integrations/gppo/search_viewpoints.py',
             'integrations/gppo/safety_layer.py', str(scenario.relative_to(PROJECT)),
             'configs/scenarios/assets/extended_connected_gppo_20261010.png']
    result = dict(base_head='5692ca4fe5c56f398707deea87c1efc844297b71',
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        python=platform.python_version(), torch=benchmark.torch.__version__,
        numpy=benchmark.np.__version__, device='cpu', cases=reports,
        base_sha256=base_sha, candidate_sha256=candidate_sha, marvel_sha256=marvel_sha,
        source_sha256={f: sha(PROJECT/f) for f in files},
        purpose='decision-space discovery only; known development seeds; no performance acceptance')
    (args.out/'manifest.json').write_text(json.dumps(result, indent=2))
    if args.verify_passive:
        # One additional short known case, same policy with recorder disabled.
        enabled = args.out/reports[0]['case']/'gppo'
        disabled = args.out/'passive_disabled'
        disabled.mkdir()
        benchmark.SimulationRuntime = Runtime
        benchmark.MARVELPolicyAdapter = adapter_type
        benchmark.load_and_validate_scenario = lambda file: load_domain(loader, file, scenario, False, .2)
        result = benchmark.run(disabled, 'gppo', 30, 1280021, args.steps,
                               physical_coverage=True, stop_on_completion=False,
                               wall_budget_seconds=args.wall_seconds_per_case)
        proof = verify_passivity(enabled, disabled/'gppo')
        proof.update(control_steps=args.steps, uavs=30, seed=1280021)
        (args.out/'passivity.json').write_text(json.dumps(proof, indent=2))
        print('PASSIVE_TRAJECTORY_EQUIVALENCE_OK', flush=True)

if __name__ == '__main__':
    main()
