"""System 1 (RSSM) and System 2 (logic-integrated neural network)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class TransitionModel(nn.Module):
    """Recurrent state-space model (RSSM) transition/posterior model."""

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        action_size: int,
        hidden_size: int,
        embedding_size: int,
        activation_function: str = "elu",
        min_std_dev: float = 0.1,
    ):
        super().__init__()
        self.belief_size = belief_size
        self.state_size = state_size
        self.action_size = action_size
        self.min_std_dev = min_std_dev
        self.act_fn = getattr(F, activation_function)

        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    def forward(
        self,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_belief: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None,
    ):
        """Roll the RSSM over a time-major action sequence.

        Parameters
        ----------
        prev_state, prev_belief : [B, *]
        actions : [T, B, A]
        observations : optional [T, B, E]
            Embeddings of the observations reached after each action.
        nonterminals : optional [T, B, 1]
        """
        beliefs = []
        prior_states, prior_means, prior_std_devs = [], [], []
        posterior_states, posterior_means, posterior_std_devs = [], [], []

        state = prev_state
        belief = prev_belief
        for t in range(actions.size(0)):
            if nonterminals is not None:
                state = state * nonterminals[t]
                belief = belief * nonterminals[t]

            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([state, actions[t]], dim=-1)))
            belief = self.rnn(hidden, belief)

            prior_hidden = self.act_fn(self.fc_embed_belief_prior(belief))
            prior_mean, prior_raw_std = torch.chunk(self.fc_state_prior(prior_hidden), 2, dim=-1)
            prior_std = F.softplus(prior_raw_std) + self.min_std_dev
            prior_state = prior_mean + prior_std * torch.randn_like(prior_mean)

            beliefs.append(belief)
            prior_states.append(prior_state)
            prior_means.append(prior_mean)
            prior_std_devs.append(prior_std)

            if observations is not None:
                post_hidden = self.act_fn(
                    self.fc_embed_belief_posterior(torch.cat([belief, observations[t]], dim=-1))
                )
                post_mean, post_raw_std = torch.chunk(self.fc_state_posterior(post_hidden), 2, dim=-1)
                post_std = F.softplus(post_raw_std) + self.min_std_dev
                post_state = post_mean + post_std * torch.randn_like(post_mean)
                posterior_states.append(post_state)
                posterior_means.append(post_mean)
                posterior_std_devs.append(post_std)
                state = post_state
            else:
                state = prior_state

        out = [
            torch.stack(beliefs, dim=0),
            torch.stack(prior_states, dim=0),
            torch.stack(prior_means, dim=0),
            torch.stack(prior_std_devs, dim=0),
        ]
        if observations is not None:
            out += [
                torch.stack(posterior_states, dim=0),
                torch.stack(posterior_means, dim=0),
                torch.stack(posterior_std_devs, dim=0),
            ]
        return out


class MultimodalEncoder(nn.Module):
    """Encode ray-tracing features, vehicle locations and CAoI."""

    def __init__(
        self,
        num_v: int,
        trace_feature_dim: int,
        embedding_size: int = 384,
        branch_size: int = 256,
    ):
        super().__init__()
        self.num_v = num_v
        self.trace_feature_dim = trace_feature_dim
        trace_dim = num_v * (num_v + 1) * trace_feature_dim
        loc_dim = num_v * 3
        caoi_dim = num_v

        self.trace_enc = nn.Sequential(
            nn.Linear(trace_dim, 512), nn.LayerNorm(512), nn.ELU(),
            nn.Linear(512, branch_size), nn.ELU(),
        )
        self.loc_enc = nn.Sequential(
            nn.Linear(loc_dim, 128), nn.LayerNorm(128), nn.ELU(),
            nn.Linear(128, branch_size), nn.ELU(),
        )
        self.caoi_enc = nn.Sequential(
            nn.Linear(caoi_dim, 128), nn.LayerNorm(128), nn.ELU(),
            nn.Linear(128, branch_size), nn.ELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * branch_size, embedding_size), nn.LayerNorm(embedding_size), nn.ELU()
        )

    def forward(self, trace: torch.Tensor, loc: torch.Tensor, caoi: torch.Tensor) -> torch.Tensor:
        lead = trace.shape[:-3]
        trace_flat = trace.reshape(*lead, -1)
        loc_flat = loc.reshape(*lead, -1)
        caoi_flat = caoi.reshape(*lead, -1)
        zt = self.trace_enc(trace_flat)
        zl = self.loc_enc(loc_flat)
        za = self.caoi_enc(caoi_flat)
        return self.fusion(torch.cat([zt, zl, za], dim=-1))


class MultimodalDecoder(nn.Module):
    """Decode the latent RSSM state back to normalized wireless observations."""

    def __init__(
        self,
        num_v: int,
        trace_feature_dim: int,
        belief_size: int,
        state_size: int,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.num_v = num_v
        self.trace_feature_dim = trace_feature_dim
        in_dim = belief_size + state_size
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_size), nn.LayerNorm(hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
        )
        self.trace_head = nn.Linear(hidden_size, num_v * (num_v + 1) * trace_feature_dim)
        self.loc_head = nn.Linear(hidden_size, num_v * 3)
        self.caoi_head = nn.Linear(hidden_size, num_v)

    def forward(self, belief: torch.Tensor, state: torch.Tensor):
        lead = belief.shape[:-1]
        h = self.shared(torch.cat([belief, state], dim=-1))
        trace = self.trace_head(h).reshape(*lead, self.num_v, self.num_v + 1, self.trace_feature_dim)
        loc = self.loc_head(h).reshape(*lead, self.num_v, 3)
        caoi = self.caoi_head(h).reshape(*lead, self.num_v)
        return trace, loc, caoi


class RewardModel(nn.Module):
    def __init__(self, belief_size: int, state_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, belief, state):
        return self.net(torch.cat([belief, state], dim=-1)).squeeze(-1)


class ValueModel(nn.Module):
    def __init__(self, belief_size: int, state_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, belief, state):
        return self.net(torch.cat([belief, state], dim=-1)).squeeze(-1)


class ActorModel(nn.Module):
    """Differentiable link-scheduling policy with a deterministic decoder."""

    def __init__(self, belief_size: int, state_size: int, hidden_size: int, num_v: int):
        super().__init__()
        self.V = num_v
        # Per-receiver matching classes:
        # 0=IDLE, 1=RSU, 2..V+1=vehicle senders 1..V.
        self.S = num_v + 2
        self.action_size = self.V + self.V * self.S

        self.backbone = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
        )
        self.sender_head = nn.Linear(hidden_size, self.V)
        self.match_head = nn.Linear(hidden_size + self.V, self.V * self.S)

    def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        h = self.backbone(torch.cat([belief, state], dim=-1))
        sender_logits = self.sender_head(h)
        match_logits = self.match_head(torch.cat([h, sender_logits], dim=-1))
        return torch.cat([sender_logits, match_logits], dim=-1)

    @torch.no_grad()
    def decode_schedule(self, logits: torch.Tensor) -> torch.Tensor:
        """Map one continuous action vector to a valid discrete link schedule."""
        if logits.ndim != 1:
            raise ValueError("decode_schedule expects a single 1-D action vector")
        sender_logits = logits[: self.V]
        match_logits = logits[self.V :].view(self.V, self.S)
        nominated = sender_logits > 0.0

        # Classes: IDLE and RSU are always available. Vehicle classes are
        # available only when nominated as transmitters.
        available = torch.cat(
            [
                torch.ones(2, dtype=torch.bool, device=logits.device),
                nominated.clone(),
            ],
            dim=0,
        )
        schedule = torch.full((self.V,), -1, dtype=torch.long, device=logits.device)

        for rx in range(self.V):
            if nominated[rx]:
                # A nominated vehicle is transmitting and therefore cannot receive.
                continue

            valid = available.clone()
            valid[rx + 2] = False  # no vehicle self-link
            row = match_logits[rx].clone()
            row[~valid] = -1e9
            cls = int(torch.argmax(row).item())

            if cls == 0:       # explicit IDLE choice
                schedule[rx] = -1
            elif cls == 1:     # RSU
                schedule[rx] = 0
            else:              # vehicle sender
                sender = cls - 1
                schedule[rx] = sender
                available[cls] = False
        return schedule

    # Backwards-compatible name used by the earlier project.
    def get_action(self, logits, det=False):
        if logits.ndim == 2:
            if logits.size(0) != 1:
                raise ValueError("get_action can decode only a single online action")
            logits = logits[0]
        return self.decode_schedule(logits)


class LogicIntegratedNetwork(nn.Module):
    """Logic-driven System 2 with learned NOT/AND/OR/IMPLY operators.

    The state given to System 2 is s_t = [h_t, z_t], i.e. the concatenation of
    the RSSM deterministic belief and stochastic state.  A sequence loss
    recursively conjoins premises up to ``reasoning_depth`` before applying the
    implication operator, implementing the long-horizon logic signal used to
    regularize System 1 imagination.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        logic_dim: int = 64,
        hidden_dim: int = 128,
        reasoning_depth: int = 30,
        activation: str = "relu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logic_dim = logic_dim
        self.reasoning_depth = reasoning_depth
        act = nn.ReLU if activation == "relu" else nn.ELU

        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), act(), nn.Linear(hidden_dim, logic_dim)
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), act(), nn.Linear(hidden_dim, logic_dim)
        )
        self.not_net = nn.Sequential(
            nn.Linear(logic_dim, hidden_dim), act(), nn.Linear(hidden_dim, logic_dim)
        )
        self.and_net = nn.Sequential(
            nn.Linear(2 * logic_dim, hidden_dim), act(), nn.Linear(hidden_dim, logic_dim)
        )
        self.or_net = nn.Sequential(
            nn.Linear(2 * logic_dim, hidden_dim), act(), nn.Linear(hidden_dim, logic_dim)
        )

        true = torch.randn(logic_dim)
        true = true / true.norm().clamp_min(1e-6)
        self.register_buffer("true", true)

    def logic_not(self, x):
        return self.not_net(x)

    def logic_and(self, x, y):
        # Symmetrization enforces order independence directly.
        return 0.5 * (
            self.and_net(torch.cat([x, y], dim=-1))
            + self.and_net(torch.cat([y, x], dim=-1))
        )

    def logic_or(self, x, y):
        return 0.5 * (
            self.or_net(torch.cat([x, y], dim=-1))
            + self.or_net(torch.cat([y, x], dim=-1))
        )

    def imply(self, premise, conclusion):
        return self.logic_or(self.logic_not(premise), conclusion)

    @staticmethod
    def similarity(x, y):
        # Paper uses sigmoid(cosine similarity).
        return torch.sigmoid(F.cosine_similarity(x, y, dim=-1))

    def _truth_like(self, x):
        return self.true.view(*([1] * (x.ndim - 1)), -1).expand_as(x)

    def regularization_loss(self, samples: torch.Tensor) -> torch.Tensor:
        """Differentiable versions of the classical logic identities in Table I."""
        x = samples
        T = self._truth_like(x)
        Fv = self.logic_not(T)
        nx = self.logic_not(x)

        losses = []
        # NOT
        losses += [1.0 - self.similarity(self.logic_not(nx), x)]
        # AND
        losses += [1.0 - self.similarity(self.logic_and(x, T), x)]
        losses += [1.0 - self.similarity(self.logic_and(x, Fv), Fv)]
        losses += [1.0 - self.similarity(self.logic_and(x, x), x)]
        losses += [1.0 - self.similarity(self.logic_and(x, nx), Fv)]
        # OR
        losses += [1.0 - self.similarity(self.logic_or(x, Fv), x)]
        losses += [1.0 - self.similarity(self.logic_or(x, T), T)]
        losses += [1.0 - self.similarity(self.logic_or(x, x), x)]
        losses += [1.0 - self.similarity(self.logic_or(x, nx), T)]
        # IMPLY
        losses += [1.0 - self.similarity(self.imply(x, T), T)]
        losses += [1.0 - self.similarity(self.imply(x, Fv), nx)]
        losses += [1.0 - self.similarity(self.imply(x, x), T)]
        losses += [1.0 - self.similarity(self.imply(x, nx), nx)]
        return torch.stack([term.mean() for term in losses]).mean()

    def sequence_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        regularization_weight: float = 0.1,
        include_regularization: bool = True,
    ):
        """Compute deep recursive implication loss over [T,B,*] sequences."""
        if states.size(0) == 0:
            zero = states.sum() * 0.0
            return zero, zero, {"true_similarity": 0.0, "false_similarity": 0.0}

        s = self.state_embed(states)
        a = self.action_embed(actions)
        ns = self.state_embed(next_states)
        eta = self.logic_and(s, a)

        phis = []
        for t in range(eta.size(0)):
            premise = eta[t]
            start = max(0, t - self.reasoning_depth)
            for k in range(t - 1, start - 1, -1):
                premise = self.logic_and(eta[k], premise)
            phis.append(self.imply(premise, ns[t]))
        phi = torch.stack(phis, dim=0)

        Tvec = self._truth_like(phi)
        Fvec = self.logic_not(Tvec)
        sim_t = self.similarity(phi, Tvec)
        sim_f = self.similarity(phi, Fvec)

        # Eq. (15) is written as a similarity margin. For minimization we use
        # its equivalent loss form: make implications true and unlike false.
        transition_loss = ((1.0 - sim_t) + sim_f).mean()

        reg_loss = phi.new_zeros(())
        if include_regularization:
            # Include state/action embeddings as regularization samples.
            samples = torch.cat([s.reshape(-1, self.logic_dim), a.reshape(-1, self.logic_dim)], dim=0)
            if samples.size(0) > 4096:
                samples = samples[torch.randperm(samples.size(0), device=samples.device)[:4096]]
            reg_loss = self.regularization_loss(samples)

        total = transition_loss + regularization_weight * reg_loss
        stats = {
            "true_similarity": float(sim_t.detach().mean().cpu()),
            "false_similarity": float(sim_f.detach().mean().cpu()),
        }
        return total, reg_loss, stats


# Alias matching the name used by the uploaded Dual-Mind implementation.
NLR = LogicIntegratedNetwork
