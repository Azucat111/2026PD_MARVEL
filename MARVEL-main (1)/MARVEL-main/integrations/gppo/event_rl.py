"""Double DQN and discrete SAC over the legal rows of a public TaskGraph."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .event_rl_replay import Decision, FEATURES

FORMAT = "joint-event-rl-v1"


def padded(states):
    if not states or any(not s.candidates for s in states):
        raise ValueError("An RL state needs a nonempty legal candidate set")
    size = max(len(s.candidates) for s in states)
    width = len(states[0].features[0])
    values = torch.zeros((len(states), size, width))
    legal = torch.zeros((len(states), size), dtype=torch.bool)
    for i, state in enumerate(states):
        matrix = torch.tensor(state.features, dtype=values.dtype)
        if not torch.isfinite(matrix).all() or tuple(matrix.shape) != (len(state.candidates), width):
            raise ValueError("Malformed legal action features")
        values[i, :len(matrix)] = matrix
        legal[i, :len(matrix)] = True
    return values, legal


def masked_argmax(scores, legal):
    if scores.shape != legal.shape or not bool(legal.any(-1).all()):
        raise ValueError("Cannot choose from an empty or inconsistent legal mask")
    return scores.masked_fill(~legal, -torch.inf).argmax(-1)


def masked_probabilities(scores, legal):
    masked_argmax(scores, legal)  # Reject empty rows before log-softmax.
    logp = scores.masked_fill(~legal, -torch.inf).log_softmax(-1)
    return logp.exp(), logp.masked_fill(~legal, 0.)


def dqn_continuation(online, target, legal):
    a_next = masked_argmax(online, legal)
    return target.gather(1, a_next[:, None]).squeeze(1)


def bellman(reward, duration, continuation, done, truncated, *, gamma_per_second=1.):
    if not 0. < gamma_per_second <= 1. or (duration <= 0).any() or (done & truncated).any():
        raise ValueError("Invalid SMDP Bellman interval")
    if truncated.any():
        raise ValueError("A censored last decision has no observed bootstrap state")
    discount = torch.pow(gamma_per_second, duration)
    return reward + discount * torch.where(done, 0., continuation)


class CandidateNet(nn.Module):
    def __init__(self, features, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden), nn.SiLU(),
                                    nn.Linear(hidden, 1))

    def forward(self, candidates):
        return self.layers(candidates)


class TwinCritic(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.first = CandidateNet(features, hidden)
        self.second = CandidateNet(features, hidden)

    def forward(self, candidates):
        return torch.cat((self.first(candidates), self.second(candidates)), dim=-1)


@dataclass(frozen=True)
class Settings:
    algo: str
    features: int
    seed: int
    hidden: int = 64
    lr: float = .0007
    batch_size: int = 32
    gamma_per_second: float = 1.
    sync_every: int = 50
    initial_alpha: float = .035
    entropy_fraction: float = .65
    n_step: int = 1

    def validate(self):
        if self.algo not in ("ddqn", "sac") or min(self.features, self.hidden, self.batch_size, self.sync_every,
                                                   self.n_step) <= 0 or not 0. < self.gamma_per_second <= 1.:
            raise ValueError("Invalid event RL settings")
        if not math.isfinite(self.lr) or self.lr <= 0. or not 0. < self.initial_alpha <= .3:
            raise ValueError("Invalid optimizer or temperature")
        if not 0. <= self.entropy_fraction <= 1. or (self.algo == "sac" and self.n_step != 1):
            raise ValueError("Invalid entropy target or multistep configuration")


class Learner:
    def __init__(self, settings):
        settings.validate()
        self.settings = settings
        net = TwinCritic if settings.algo == "sac" else CandidateNet
        self.critic = net(settings.features, settings.hidden)
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=settings.lr)
        self.updates = 0
        if settings.algo == "sac":
            self.actor = CandidateNet(settings.features, settings.hidden)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=settings.lr)
            self.log_alpha = nn.Parameter(torch.tensor(math.log(settings.initial_alpha)))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=settings.lr / 4.)

    def sync_target(self):
        self.target.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def continuation(self, following, legal):
        target = self.target(following)
        if self.settings.algo == "ddqn":
            return dqn_continuation(self.critic(following).squeeze(-1), target.squeeze(-1), legal)
        alpha = self.log_alpha.exp()
        probs, logp = masked_probabilities(self.actor(following).squeeze(-1), legal)
        return (probs * (target.min(-1).values - alpha * logp)).sum(-1)

    def update(self, batch):
        current, legal = padded([t.state for t in batch])
        chosen = torch.tensor([t.action for t in batch])
        indices = torch.arange(len(batch))
        if (chosen < 0).any() or (chosen >= legal.shape[1]).any() or not legal[indices, chosen].all():
            raise ValueError("Replay action is masked")
        reward = torch.tensor([t.reward for t in batch])
        durations = torch.tensor([t.dt for t in batch])
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool)
        truncated = torch.tensor([t.truncated for t in batch], dtype=torch.bool)
        observed_next = [i for i, t in enumerate(batch) if not t.done and t.next_state is not None]
        if any(t.next_state is None and not (t.done or t.truncated) for t in batch):
            raise ValueError("An unfinished transition needs a real next action graph")
        following_value = torch.zeros_like(reward)
        with torch.no_grad():
            if observed_next:
                following, next_legal = padded([batch[i].next_state for i in observed_next])
                following_value[observed_next] = self.continuation(following, next_legal)
            expected = bellman(reward, durations, following_value, dones, truncated,
                               gamma_per_second=self.settings.gamma_per_second)
        prediction = self.critic(current)[indices, chosen]
        critic_loss = F.smooth_l1_loss(prediction, expected[:, None].expand_as(prediction))
        self.optimizer.zero_grad()
        critic_loss.backward()
        grad = nn.utils.clip_grad_norm_(self.critic.parameters(), 5., error_if_nonfinite=True)
        self.optimizer.step()

        metrics = dict(critic_loss=float(critic_loss), gradient_norm=float(grad),
                       target_min=float(expected.min()), target_max=float(expected.max()),
                       prediction_mean=float(prediction.detach().mean()), bootstrap_rows=len(observed_next))
        if self.settings.algo == "sac":
            metrics.update(self.update_actor(current, legal))
        self.updates += 1
        if self.updates % self.settings.sync_every == 0:
            self.sync_target()
        return metrics

    def update_actor(self, current, legal):
        q = self.critic(current).detach().min(-1).values
        probs, logp = masked_probabilities(self.actor(current).squeeze(-1), legal)
        alpha = self.log_alpha.exp().detach()
        actor_loss = (probs * (alpha * logp - q)).sum(-1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5., error_if_nonfinite=True)
        self.actor_optimizer.step()
        entropy = -(probs.detach() * logp.detach()).sum(-1)
        target_entropy = self.settings.entropy_fraction * legal.sum(-1).float().log()
        alpha_loss = (self.log_alpha * (entropy - target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(1e-4), math.log(.3))
        return dict(actor_loss=float(actor_loss), alpha=float(alpha),
                    entropy=float(entropy.mean()), entropy_target=float(target_entropy.mean()))

    def checkpoint(self, replay_sha, audit):
        payload = dict(format=FORMAT, features=FEATURES, settings=asdict(self.settings),
                       updates=self.updates, replay_sha256=replay_sha, coverage=audit["coverage"],
                       critic=self.critic.state_dict(), target=self.target.state_dict(),
                       optimizer=self.optimizer.state_dict())
        if self.settings.algo == "sac":
            payload.update(actor=self.actor.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
                           alpha_optimizer=self.alpha_optimizer.state_dict(), log_alpha=self.log_alpha.detach())
        return payload


class EventPolicy:
    def __init__(self, data):
        if data["format"] != FORMAT or data["features"] != FEATURES or data["updates"] <= 0:
            raise ValueError("Invalid event RL checkpoint")
        self.settings = Settings(**data["settings"])
        self.settings.validate()
        self.net = CandidateNet(self.settings.features, self.settings.hidden)
        weights = data["actor"] if self.settings.algo == "sac" else data["critic"]
        self.net.load_state_dict(weights, strict=True)
        self.net.requires_grad_(False).eval()

    @classmethod
    def load(cls, path: Path, sha: str):
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Event RL checkpoint hash mismatch")
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    @torch.no_grad()
    def choose(self, event):
        state = Decision.from_event(event)
        values, legal = padded([state])
        scores = self.net(values).squeeze(-1)
        index = int(masked_argmax(scores, legal)[0])
        result = dict(algo=self.settings.algo, scores=scores[0].tolist(), selected_index=index)
        if self.settings.algo == "sac":
            result["policy_probabilities"] = masked_probabilities(scores, legal)[0][0].tolist()
        return event["rows"][index]["flat"], result


def n_step_episode(rows, n, gamma):
    from dataclasses import replace

    result = []
    for i, row in enumerate(rows):
        span = rows[i:i+n]
        elapsed, reward = 0., 0.
        for entry in span:
            reward += gamma**elapsed * entry.reward
            elapsed += entry.dt
        tail = span[-1]
        result.append(replace(row, reward=reward, dt=elapsed, next_state=tail.next_state,
                              done=tail.done, truncated=tail.truncated))
    return result


def train(episodes, audit, settings, updates, replay_sha, progress):
    if updates <= 0 or not episodes:
        raise ValueError("Training requires committed episodes and positive updates")
    torch.manual_seed(settings.seed)
    generator = random.Random(settings.seed)
    learner = Learner(settings)
    grouped = defaultdict(list)
    for _, transitions in episodes:
        if not transitions or not transitions[-1].done or any(t.truncated for t in transitions):
            raise ValueError("Only complete event episodes can train")
        grouped[transitions[0].group].extend(n_step_episode(transitions, settings.n_step,
                                                           settings.gamma_per_second))
    groups = sorted(grouped)
    start = time.monotonic()
    for step in range(1, updates + 1):
        batch = [generator.choice(grouped[generator.choice(groups)]) for _ in range(settings.batch_size)]
        row = learner.update(batch)
        if step == 1 or step % 25 == 0 or step == updates:
            progress(dict(update=step, seconds=time.monotonic()-start, target_sync=step%settings.sync_every==0,
                          unique_environment_transitions=audit["coverage"]["transitions"], **row))
    return learner.checkpoint(replay_sha, audit)
