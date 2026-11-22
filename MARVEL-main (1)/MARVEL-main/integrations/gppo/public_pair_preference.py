"""A public, mask-preserving GPPO residual fit to paired physical returns.

The features only describe a legal event and its candidates. No simulated
position forecast is used as a motion command or as an admissibility test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .mesh_relay import mesh_components

FORMAT = "joint-public-pair-preference-v1"
FEATURE_VERSION = "public-mesh-candidate-v1"


def event_features(event):
    stale = {int(uid): np.asarray(xy, dtype=float) for uid, xy in event["stale_xy"].items()}
    rows = event["rows"]
    radius = float(event["range_m"])
    uids = sorted(stale)
    n = len(uids)
    coords = np.stack([stale[uid] for uid in uids])
    center = coords.mean(axis=0)
    separation = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    groups = mesh_components(stale, radius)
    search = {r["slot"] for r in rows if r["role"] == 1}
    relay = {r["slot"] for r in rows if r["role"] == 2}
    relay_anchors = np.asarray([r["anchor"] for r in rows if r["role"] == 2], dtype=float)
    sizes = np.pad(np.asarray([len(g)/n for g in groups[:3]]), (0, max(0, 3-len(groups))))
    context = np.array([event["time_seconds"]/1.6, len(search)/1.5, len(relay),
                        len(event["search_before"]), len(event["relay_before"]), n/30.,
                        len(event["unavailable"])/n*3., *(sizes*3.)], dtype=float)
    donor_groups = {}
    features = []
    for row in rows:
        uid = int(row["uav"])
        u = uids.index(uid)
        other = np.arange(n) != u
        own = stale[uid]
        anchor = np.asarray(row["anchor"])
        movement = anchor - own
        length = np.linalg.norm(movement)
        # Certificated waypoint direction, projected at most a radio-range fraction.
        prefix = np.asarray(row["waypoint"]) - own
        delta = prefix * min(1., (0.4*radius)/max(1e-9, np.linalg.norm(prefix)))
        if uid not in donor_groups:
            donor_groups[uid] = mesh_components(stale, radius, (uid,))

        def mesh_size(at):
            reached = set(np.asarray(uids)[(np.linalg.norm(coords-at, axis=1) <= radius) & other])
            components = donor_groups[uid]
            return max(len(components[0]) if components else 0,
                       1 + sum(len(g) for g in components if reached.intersection(g)))

        old_edges = (separation[u] <= radius) & other
        held_edges = np.linalg.norm(coords-(own+delta), axis=1) <= radius
        lost = np.count_nonzero(old_edges & ~held_edges) / max(1, np.count_nonzero(old_edges))
        nearest_relay = min(2., float(np.linalg.norm(relay_anchors-own, axis=1).min()/radius)) if relay_anchors.size else 0.
        features.append([2.*(row["role"]==2), min(row["total"], 100.)/15.,
                         *((own-center)/(1.8*radius)), *(movement/(1.4*radius)),
                         *(delta/(0.5*radius)), float(old_edges.sum())/n*4., lost*2.,
                         (mesh_size(own+delta)-len(groups[0]))/n*4.,
                         (mesh_size(anchor)-len(groups[0]))/n*4.,
                         min(length/radius, 4.), nearest_relay])
    vectors = np.asarray(features, dtype=float)
    if not np.isfinite(vectors).all() or not np.isfinite(context).all():
        raise ValueError("Nonfinite public ranking features")
    return context, vectors


def kernel(left_context, left, right_context, right, *, context_scale=1., action_scale=1.):
    state_sq = np.square((left_context[:, None, :]-right_context[None, :, :])/context_scale).sum(-1)
    action_sq = np.square((left[:, None, :]-right[None, :, :])/action_scale).sum(-1)
    return np.exp(-.5*(state_sq + action_sq))


def make_checkpoint(contrasts, *, base_sha, mode, context_scale, action_scale,
                    ridge=.20, prior_scale=.25, support=.45, margin=.20):
    """Solve ridge regression on differences of action evaluations in one state."""
    if not contrasts:
        raise ValueError("Need at least one verified paired contrast")
    contexts = np.repeat([row["context"] for row in contrasts], 2, axis=0)
    actions = np.asarray([feature for row in contrasts for feature in (row["new_features"], row["old_features"])])
    k = kernel(contexts, actions, contexts, actions, context_scale=context_scale, action_scale=action_scale)
    diff_gram = k[::2, ::2] - k[::2, 1::2] - k[1::2, ::2] + k[1::2, 1::2]
    target = [np.clip((row["control_seconds"]-row["treatment_seconds"])/10., -8., 8.)
              - prior_scale*(row["new_logit"]-row["old_logit"]) for row in contrasts]
    weights = np.linalg.solve(diff_gram + ridge*np.eye(len(contrasts)), target)
    return dict(format=FORMAT, feature_version=FEATURE_VERSION, base_sha256=base_sha, mode=mode,
                context_scale=context_scale, action_scale=action_scale, ridge=ridge, prior_scale=prior_scale,
                support=support, margin=margin, contexts=contexts.tolist(), actions=actions.tolist(),
                weights=np.column_stack([weights,-weights]).flatten().tolist(), physical_training_pairs=contrasts)


class PublicPairPreference:
    """Residual policy with explicit legal support and a deterministic fallback."""
    def __init__(self, checkpoint):
        self.parameters = checkpoint
        if checkpoint["format"] != FORMAT or checkpoint["feature_version"] != FEATURE_VERSION:
            raise ValueError("Incompatible ranking checkpoint")
        if checkpoint["mode"] not in ("within_role", "joint"):
            raise ValueError("Unknown ranking mode")
        self.contexts = np.asarray(checkpoint["contexts"], dtype=float)
        self.actions = np.asarray(checkpoint["actions"], dtype=float)
        self.weights = np.asarray(checkpoint["weights"], dtype=float)
        if (self.contexts.ndim != 2 or self.actions.ndim != 2 or self.weights.shape != (len(self.contexts),)
                or len(self.actions) != len(self.contexts) or not len(self.weights)
                or any(not np.isfinite(x).all() for x in (self.contexts, self.actions, self.weights))):
            raise ValueError("Malformed ranking checkpoint")
        for key in ("prior_scale", "margin", "action_scale", "context_scale", "support"):
            if not np.isfinite(checkpoint[key]) or checkpoint[key]<=0:
                raise ValueError(f"Invalid ranking {key}")

    @classmethod
    def load(cls, path: Path, expected_sha: str):
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != expected_sha:
            raise ValueError("Ranking checkpoint hash mismatch")
        return cls(json.loads(blob))

    def choose(self, event):
        p = self.parameters
        rows = event["rows"]
        context, features = event_features(event)
        similarity = kernel(np.repeat(context[None, :], len(rows), axis=0), features,
                            self.contexts, self.actions,
                            context_scale=p["context_scale"], action_scale=p["action_scale"])
        support = similarity.max(axis=1)
        logit = np.asarray([row["logit"] for row in rows]) * p["prior_scale"]
        role = np.asarray([row["role"] for row in rows])
        normal = int(logit.argmax())
        residual = np.clip(similarity @ self.weights, -8., 8.)
        scores = logit + residual
        if p["mode"] == "within_role":
            # Preserve each role's maximum while changing within-role order.
            # No candidate is removed from the distribution.
            for r in set(role):
                group = role == r
                scores[group] -= scores[group].max()-logit[group].max()
        preferred = (int(np.where(role == role[normal], scores, -np.inf).argmax())
                     if p["mode"] == "within_role" else int(scores.argmax()))
        # Evidence is required for both ends of a behavioral change.
        # Centering of a role's logits is undone when falling back to J0.
        if (min(support[normal], support[preferred]) < p["support"] or
                scores[preferred]-scores[normal] < p["margin"]):
            scores = logit
            preferred = normal
        probs = np.exp(scores-scores.max())
        probs /= probs.sum()
        return rows[preferred]["flat"], dict(mode=p["mode"], prior_index=rows[normal]["flat"],
                                             chosen_index=rows[preferred]["flat"],
                                             legal_indices=[row["flat"] for row in rows],
                                             prior_logits=logit.tolist(), logits=scores.tolist(),
                                             probs=probs.tolist(), supports=support.tolist(),
                                             residual_before_guard=residual.tolist())
