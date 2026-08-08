"""Model-training functions independent of Sionna RT.

Keeping this module free of Mitsuba/Sionna imports allows the System 1/System 2
integration to be unit-tested on synthetic tensors.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from utils import FreezeParameters, imagine_ahead, lambda_return


def posterior_batch(args, batch, transition, encoder):
    traces, locations, caoi, actions, rewards, nonterminals = batch
    batch_size = traces.size(1)
    init_state = torch.zeros(batch_size, args.state_size, device=traces.device)
    init_belief = torch.zeros(batch_size, args.belief_size, device=traces.device)
    embeddings = encoder(traces[1:], locations[1:], caoi[1:])
    return transition(
        init_state,
        actions[:-1],
        init_belief,
        observations=embeddings,
        nonterminals=nonterminals[:-1],
    )


def system1_loss(args, batch, transition, encoder, decoder, reward_model):
    traces, locations, caoi, actions, rewards, nonterminals = batch
    (
        beliefs,
        prior_states,
        prior_means,
        prior_stds,
        posterior_states,
        posterior_means,
        posterior_stds,
    ) = posterior_batch(args, batch, transition, encoder)

    rec_trace, rec_loc, rec_caoi = decoder(beliefs, posterior_states)
    trace_loss = torch.mean((rec_trace - traces[1:]) ** 2)
    loc_loss = torch.mean((rec_loc - locations[1:]) ** 2)
    caoi_loss = torch.mean((rec_caoi - caoi[1:]) ** 2)

    predicted_reward = reward_model(beliefs, posterior_states)
    reward_loss = torch.mean((predicted_reward - rewards[:-1]) ** 2)

    q_detached = Normal(posterior_means.detach(), posterior_stds.detach())
    p = Normal(prior_means, prior_stds)
    dyn_kl = kl_divergence(q_detached, p).sum(dim=-1)

    q = Normal(posterior_means, posterior_stds)
    p_detached = Normal(prior_means.detach(), prior_stds.detach())
    rep_kl = kl_divergence(q, p_detached).sum(dim=-1)

    dyn_loss = torch.clamp(dyn_kl, min=args.free_nats).mean()
    rep_loss = torch.clamp(rep_kl, min=args.free_nats).mean()
    prediction_loss = trace_loss + loc_loss + caoi_loss + reward_loss
    total = prediction_loss + args.dyn_weight * dyn_loss + args.rep_weight * rep_loss

    stats = {
        "model_loss": total,
        "trace_loss": trace_loss,
        "location_loss": loc_loss,
        "caoi_loss": caoi_loss,
        "reward_loss": reward_loss,
        "dyn_loss": dyn_loss,
        "rep_loss": rep_loss,
    }
    return total, stats


def real_logic_sequences(args, batch, transition, encoder):
    with torch.no_grad():
        out = posterior_batch(args, batch, transition, encoder)
        beliefs, posterior_states = out[0], out[4]
        latent = torch.cat([beliefs, posterior_states], dim=-1)
    actions = batch[3]
    return latent[:-1].detach(), actions[1:-1].detach(), latent[1:].detach()


def train_one_update(
    args,
    batch,
    transition,
    encoder,
    decoder,
    reward_model,
    actor,
    value_model,
    logic_model,
    model_optimizer,
    logic_optimizer,
    model_logic_optimizer,
    actor_optimizer,
    value_optimizer,
):
    # 1) System 1 learns statistical dynamics and reconstruction/reward.
    model_optimizer.zero_grad(set_to_none=True)
    loss, s1 = system1_loss(args, batch, transition, encoder, decoder, reward_model)
    loss.backward()
    nn.utils.clip_grad_norm_(
        list(transition.parameters()) + list(encoder.parameters())
        + list(decoder.parameters()) + list(reward_model.parameters()),
        args.grad_clip_norm,
    )
    model_optimizer.step()

    # 2) System 2 learns logical transition rules from real posterior states.
    logic_states, logic_actions, logic_next = real_logic_sequences(args, batch, transition, encoder)
    logic_optimizer.zero_grad(set_to_none=True)
    logic_loss, logic_reg, logic_stats = logic_model.sequence_loss(
        logic_states,
        logic_actions,
        logic_next,
        regularization_weight=args.logic_reg_weight,
        include_regularization=True,
    )
    logic_loss.backward()
    nn.utils.clip_grad_norm_(logic_model.parameters(), args.grad_clip_norm)
    logic_optimizer.step()

    # 3) Inter-system signal: freeze System 2 and the actor, then backpropagate
    #    logical inconsistency through imagined states into the RSSM transition.
    with torch.no_grad():
        out = posterior_batch(args, batch, transition, encoder)
        start_beliefs, start_states = out[0].detach(), out[4].detach()

    model_logic_optimizer.zero_grad(set_to_none=True)
    with FreezeParameters([actor, logic_model]):
        imag_logic = imagine_ahead(
            start_states,
            start_beliefs,
            actor,
            transition,
            planning_horizon=args.planning_horizon,
            max_starts=args.imagination_starts,
            detach_actions=True,
        )
        imagined_s = torch.cat([imag_logic.beliefs[:-1], imag_logic.states[:-1]], dim=-1)
        imagined_next = torch.cat([imag_logic.beliefs[1:], imag_logic.states[1:]], dim=-1)
        guidance_loss, _, _ = logic_model.sequence_loss(
            imagined_s,
            imag_logic.actions,
            imagined_next,
            regularization_weight=0.0,
            include_regularization=False,
        )
    (args.logic_guidance_weight * guidance_loss).backward()
    nn.utils.clip_grad_norm_(transition.parameters(), args.grad_clip_norm)
    model_logic_optimizer.step()

    # 4) Actor learns in differentiable imagined trajectories.
    with torch.no_grad():
        out = posterior_batch(args, batch, transition, encoder)
        actor_beliefs, actor_states = out[0].detach(), out[4].detach()

    actor_optimizer.zero_grad(set_to_none=True)
    with FreezeParameters([transition, reward_model, value_model]):
        imag = imagine_ahead(
            actor_states,
            actor_beliefs,
            actor,
            transition,
            planning_horizon=args.planning_horizon,
            max_starts=args.imagination_starts,
            detach_actions=False,
        )
        imagined_reward = reward_model(imag.next_beliefs, imag.next_states)
        value_pred = value_model(imag.next_beliefs, imag.next_states)
        returns = lambda_return(
            imagined_reward,
            value_pred,
            bootstrap=value_pred[-1],
            discount=args.discount,
            lambda_=args.return_lambda,
        )
        actor_loss = -returns.mean()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip_norm)
    actor_optimizer.step()

    # 5) Critic learns lambda returns from imagination.
    value_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        vb = imag.next_beliefs.detach()
        vs = imag.next_states.detach()
        target = returns.detach()
    value_pred_train = value_model(vb, vs)
    value_loss = -Normal(value_pred_train, 1.0).log_prob(target).mean()
    value_loss.backward()
    nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm)
    value_optimizer.step()

    return {
        **{k: float(v.detach().cpu()) for k, v in s1.items()},
        "logic_loss": float(logic_loss.detach().cpu()),
        "logic_regularization": float(logic_reg.detach().cpu()),
        "logic_guidance_loss": float(guidance_loss.detach().cpu()),
        "logic_true_similarity": logic_stats["true_similarity"],
        "logic_false_similarity": logic_stats["false_similarity"],
        "actor_loss": float(actor_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
    }
