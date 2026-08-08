"""Training utilities for the dual-mind world model."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Imagination:
    beliefs: torch.Tensor       # [H+1, N, belief]
    states: torch.Tensor        # [H+1, N, state]
    actions: torch.Tensor       # [H,   N, action]
    prior_means: torch.Tensor   # [H,   N, state]
    prior_stds: torch.Tensor    # [H,   N, state]

    @property
    def next_beliefs(self):
        return self.beliefs[1:]

    @property
    def next_states(self):
        return self.states[1:]


@contextmanager
def freeze_parameters(modules: Iterable[nn.Module]):
    params = []
    states = []
    for module in modules:
        for p in module.parameters():
            params.append(p)
            states.append(p.requires_grad)
            p.requires_grad_(False)
    try:
        yield
    finally:
        for p, state in zip(params, states):
            p.requires_grad_(state)


# Backwards-compatible class-style context manager.
class FreezeParameters:
    def __init__(self, modules):
        self.modules = list(modules)
        self._ctx = None

    def __enter__(self):
        self._ctx = freeze_parameters(self.modules)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)


def imagine_ahead(
    prev_state: torch.Tensor,
    prev_belief: torch.Tensor,
    policy,
    transition_model,
    planning_horizon: int = 30,
    max_starts: int | None = None,
    detach_actions: bool = False,
) -> Imagination:
    """Roll out differentiable latent trajectories under the actor.

    ``prev_state`` and ``prev_belief`` may be [T,B,D] posterior trajectories or
    a flat [N,D] set of starting points. The first two dimensions are flattened
    so that imagination can start from many real posterior states at once.
    """
    if prev_state.ndim > 2:
        state = prev_state.reshape(-1, prev_state.size(-1))
        belief = prev_belief.reshape(-1, prev_belief.size(-1))
    else:
        state = prev_state
        belief = prev_belief

    if max_starts is not None and state.size(0) > max_starts:
        idx = torch.randperm(state.size(0), device=state.device)[:max_starts]
        state = state[idx]
        belief = belief[idx]

    all_states = [state]
    all_beliefs = [belief]
    actions, means, stds = [], [], []

    for _ in range(planning_horizon):
        action = policy(belief, state)
        action_for_dynamics = action.detach() if detach_actions else action

        hidden = transition_model.act_fn(
            transition_model.fc_embed_state_action(torch.cat([state, action_for_dynamics], dim=-1))
        )
        belief = transition_model.rnn(hidden, belief)
        prior_hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(belief))
        mean, raw_std = torch.chunk(transition_model.fc_state_prior(prior_hidden), 2, dim=-1)
        std = F.softplus(raw_std) + transition_model.min_std_dev
        state = mean + std * torch.randn_like(mean)

        actions.append(action_for_dynamics)
        means.append(mean)
        stds.append(std)
        all_beliefs.append(belief)
        all_states.append(state)

    return Imagination(
        beliefs=torch.stack(all_beliefs, dim=0),
        states=torch.stack(all_states, dim=0),
        actions=torch.stack(actions, dim=0),
        prior_means=torch.stack(means, dim=0),
        prior_stds=torch.stack(stds, dim=0),
    )


def lambda_return(
    imagined_reward: torch.Tensor,
    value_pred: torch.Tensor,
    bootstrap: torch.Tensor,
    discount: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Generalized lambda return for time-major tensors [H,N]."""
    next_values = torch.cat([value_pred[1:], bootstrap.unsqueeze(0)], dim=0)
    inputs = imagined_reward + discount * next_values * (1.0 - lambda_)
    last = bootstrap
    outputs = []
    for t in reversed(range(inputs.size(0))):
        last = inputs[t] + discount * lambda_ * last
        outputs.append(last)
    return torch.stack(list(reversed(outputs)), dim=0)


def grad_norm(parameters) -> float:
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).cpu())
