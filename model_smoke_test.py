"""CPU-only smoke test for System 1/System 2 integration.

This file intentionally does not import Sionna RT. It verifies that one complete
Dual-Mind optimization update (System 1 -> System 2 -> logic guidance -> actor
-> critic) runs and that the discrete scheduler supports an explicit IDLE action.
"""
from types import SimpleNamespace

import torch

from models import (
    ActorModel,
    LogicIntegratedNetwork,
    MultimodalDecoder,
    MultimodalEncoder,
    RewardModel,
    TransitionModel,
    ValueModel,
)
from trainer import train_one_update


def main():
    torch.manual_seed(7)
    args = SimpleNamespace(
        num_v=4,
        state_size=32,
        belief_size=32,
        hidden_size=64,
        embedding_size=48,
        free_nats=1.0,
        dyn_weight=1.0,
        rep_weight=1.0,
        logic_reg_weight=0.1,
        logic_guidance_weight=1.0,
        planning_horizon=5,
        imagination_starts=8,
        discount=0.99,
        return_lambda=0.95,
        grad_clip_norm=100.0,
    )
    trace_features = 7
    actor = ActorModel(args.belief_size, args.state_size, args.hidden_size, args.num_v)
    action_size = actor.action_size
    transition = TransitionModel(
        args.belief_size, args.state_size, action_size, args.hidden_size, args.embedding_size
    )
    encoder = MultimodalEncoder(args.num_v, trace_features, args.embedding_size)
    decoder = MultimodalDecoder(args.num_v, trace_features, args.belief_size, args.state_size)
    reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size)
    value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size)
    logic_model = LogicIntegratedNetwork(
        args.belief_size + args.state_size,
        action_size,
        logic_dim=16,
        reasoning_depth=4,
    )

    model_optimizer = torch.optim.Adam(
        list(transition.parameters())
        + list(encoder.parameters())
        + list(decoder.parameters())
        + list(reward_model.parameters()),
        lr=1e-3,
    )
    logic_optimizer = torch.optim.SGD(logic_model.parameters(), lr=1e-2)
    model_logic_optimizer = torch.optim.Adam(transition.parameters(), lr=1e-3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-4)

    T, B = 10, 3
    batch = (
        torch.randn(T, B, args.num_v, args.num_v + 1, trace_features),
        torch.randn(T, B, args.num_v, 3),
        torch.rand(T, B, args.num_v),
        torch.randn(T, B, action_size),
        torch.randn(T, B),
        torch.ones(T, B, 1),
    )

    stats = train_one_update(
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
    )

    required = (
        "model_loss",
        "logic_loss",
        "logic_guidance_loss",
        "actor_loss",
        "value_loss",
    )
    assert all(torch.isfinite(torch.tensor(stats[k])) for k in required)

    # Explicit IDLE action: all vehicles should remain idle.
    logits = torch.full((action_size,), -4.0)
    S = args.num_v + 2
    offset = args.num_v
    for rx in range(args.num_v):
        logits[offset + rx * S] = 4.0  # matching class 0 = IDLE
    schedule = actor.decode_schedule(logits)
    assert torch.equal(schedule, torch.full_like(schedule, -1))

    print("MODEL_SMOKE_TEST_OK")
    print("action_size:", action_size)
    for key in required:
        print(f"{key}: {stats[key]:.6f}")


if __name__ == "__main__":
    main()
