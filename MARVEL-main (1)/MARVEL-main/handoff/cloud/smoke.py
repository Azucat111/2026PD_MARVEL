"""Validate the handoff's model assets, scenario and one fixed legal action."""
import hashlib
from pathlib import Path
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts")]

from integrations.gppo.checkpoint import load_frozen_gppo
from integrations.gppo.gppo_model import GPPOActorCritic
from integrations.gppo.heading_adapter import ExtendedGPPOInferenceAdapter
from integrations.rule_scheduling import choose_rule_assignment
from test_parameter import EMBEDDING_DIM, NODE_INPUT_DIM, NUM_ANGLES_BIN
from utils.model import PolicyNet
from utils.scenario_config import load_and_validate_scenario

def main():
    manifest = json.loads((HERE / "checkpoint_manifest.json").read_text())
    paths = {}
    for row in manifest:
        path = PROJECT / row["cloud_path"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row["cloud_sha256"]
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        paths[row["name"]] = path
        if row["name"] == "MARVEL low-level actor":
            PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).load_state_dict(
                checkpoint["policy_model"], strict=True)
        else:
            model = GPPOActorCritic(edge_feature_dim=row["input_dimensions"]["edge"])
            model.load_state_dict(checkpoint["model"], strict=True)
    base = paths["Frozen native GPPO"]
    load_frozen_gppo(base)
    adapter = ExtendedGPPOInferenceAdapter(base, paths["November 11 guarded candidate"])
    logits = adapter.model(
        uav_features=torch.zeros(1, 2, 13), task_features=torch.zeros(1, 1, 12),
        edge_features=torch.zeros(1, 1, 2, 6),
        capability_matrix=torch.ones(1, 1, 2, dtype=torch.bool),
        precedence_matrix=torch.zeros(1, 1, 1, dtype=torch.bool),
        action_mask=torch.tensor([[[False, True]]]),
    )["flat_logits"]
    assert logits.argmax(dim=-1).item() == 0
    scenario = PROJECT / "configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml"
    config = load_and_validate_scenario(scenario)
    assert Path(config["environment"]["map_file"]).is_file()
    assert config["scenario"]["duration"]["step_dt"] == 0.1
    print(f"SMOKE_OK models={len(paths)} finite_legal_logit={bool(np.isfinite(logits[0, 0].item()))}")

if __name__ == "__main__":
    main()
