"""Bounded CPU public-state sampling/replay; no training or H300 acceptance."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--wall-seconds-per-case', type=float, default=180)
    p.add_argument('--replay', type=Path)
    p.add_argument('--verify-passive', action='store_true')
    args = p.parse_args()
    if not 2 <= args.steps <= 20:
        p.error('Discovery smoke is bounded to 2..20 control steps')
    args.out.mkdir(parents=True, exist_ok=False)
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
        for file in ('initial.json', 'config.json', 'decisions.json', 'events.json', 'collisions.json', 'progress.csv'):
            assert (enabled/file).read_bytes() == (disabled/'gppo'/file).read_bytes(), file
        (args.out/'passivity.json').write_text(json.dumps(dict(exact_files_equal=True,
            compared=['initial', 'config', 'decisions', 'events', 'collisions', 'progress'],
            control_steps=args.steps, uavs=30, seed=1280021), indent=2))
        print('PASSIVE_TRAJECTORY_EQUIVALENCE_OK', flush=True)

if __name__ == '__main__':
    main()
