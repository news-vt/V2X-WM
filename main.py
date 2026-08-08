"""Train the integrated Dual-Mind World Model on the Sionna V2X environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

from runtime import configure_drjit_llvm

# Critical on macOS: configure LLVM before env.py imports Mitsuba/Sionna RT.
configure_drjit_llvm()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn, optim  # noqa: E402
from tqdm import tqdm  # noqa: E402

from env import SionnaV2XEnv  # noqa: E402
from memory import ExperienceReplay  # noqa: E402
from models import (  # noqa: E402
    ActorModel,
    LogicIntegratedNetwork,
    MultimodalDecoder,
    MultimodalEncoder,
    RewardModel,
    TransitionModel,
    ValueModel,
)
from trainer import train_one_update  # noqa: E402

try:  # tensorboard is useful but should not prevent training from starting
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    class SummaryWriter:  # type: ignore
        def __init__(self, *_, **__): pass
        def add_scalar(self, *_, **__): pass
        def close(self): pass


def build_parser():
    p = argparse.ArgumentParser("Dual-Mind World Model for dynamic mmWave V2X")

    # Environment: defaults follow Table II in the paper.
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--run-id", type=str, default="1")
    p.add_argument("--scene", type=str, default="itu_scene/itu_scene.xml")
    p.add_argument("--num-v", type=int, default=8)
    p.add_argument("--num-antennas", type=int, default=4)
    p.add_argument("--road-length", type=float, default=200.0)
    p.add_argument("--min-gap", type=float, default=20.0)
    p.add_argument("--tx-power-dbm", type=float, default=23.0)
    p.add_argument("--bandwidth", type=float, default=100e6)
    p.add_argument("--frequency", type=float, default=26e9)
    p.add_argument("--slot-duration", type=float, default=0.1)
    p.add_argument("--packet-size-bytes", type=float, default=5e6)
    p.add_argument("--num-packets", type=int, default=25)
    p.add_argument("--caoi-tolerance", type=float, default=8.0)
    p.add_argument("--max-episode-length", type=int, default=100)
    p.add_argument("--noise-figure-db", type=float, default=0.0)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--max-num-paths", type=int, default=4000)
    p.add_argument("--samples-per-src", type=int, default=20000)
    p.add_argument("--disable-diffuse", action="store_true")
    p.add_argument("--explicit-array", action="store_true")
    p.add_argument("--terminate-on-tolerance", action="store_true")

    # System 1.
    p.add_argument("--embedding-size", type=int, default=384)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--belief-size", type=int, default=256)
    p.add_argument("--state-size", type=int, default=256)
    p.add_argument("--free-nats", type=float, default=1.0)
    p.add_argument("--dyn-weight", type=float, default=1.0)
    p.add_argument("--rep-weight", type=float, default=1.0)

    # System 2.
    p.add_argument("--logic-vector-size", type=int, default=64)
    p.add_argument("--reasoning-depth", type=int, default=30)
    p.add_argument("--logic-reg-weight", type=float, default=0.1)
    p.add_argument("--logic-guidance-weight", type=float, default=10.0)
    p.add_argument("--logic-learning-rate", type=float, default=1e-2)
    p.add_argument("--model-logic-learning-rate", type=float, default=1e-3)

    # Actor-critic / training.
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed-episodes", type=int, default=5)
    p.add_argument("--collect-interval", type=int, default=100,
                   help="Gradient updates per environment episode")
    p.add_argument("--experience-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--planning-horizon", type=int, default=30)
    p.add_argument("--imagination-starts", type=int, default=256)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--return-lambda", type=float, default=0.95)
    p.add_argument("--action-noise", type=float, default=0.3)
    p.add_argument("--model-learning-rate", type=float, default=1e-3)
    p.add_argument("--actor-learning-rate", type=float, default=1e-4)
    p.add_argument("--value-learning-rate", type=float, default=1e-4)
    p.add_argument("--adam-epsilon", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=100.0)

    # Evaluation / deployment-style missing observations.
    p.add_argument("--test", action="store_true")
    p.add_argument("--test-episodes", type=int, default=10)
    p.add_argument("--test-interval", type=int, default=25)
    p.add_argument("--prediction-drop-prob", type=float, default=0.0,
                   help="Probability of omitting a real observation during online control")

    p.add_argument("--checkpoint-interval", type=int, default=25)
    p.add_argument("--models", type=str, default="")
    p.add_argument("--checkpoint-experience", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu", "cuda"])
    return p


def make_env(args):
    return SionnaV2XEnv(
        scene_path=args.scene,
        max_episode_length=args.max_episode_length,
        num_v=args.num_v,
        num_antennas=args.num_antennas,
        road_length=args.road_length,
        min_gap=args.min_gap,
        tx_power_dbm=args.tx_power_dbm,
        bandwidth=args.bandwidth,
        carrier_frequency=args.frequency,
        slot_duration=args.slot_duration,
        packet_size_bytes=args.packet_size_bytes,
        num_packets=args.num_packets,
        caoi_tolerance=args.caoi_tolerance,
        seed=args.seed,
        noise_figure_db=args.noise_figure_db,
        max_depth=args.max_depth,
        max_num_paths_per_src=args.max_num_paths,
        samples_per_src=args.samples_per_src,
        diffuse_reflection=not args.disable_diffuse,
        synthetic_array=not args.explicit_array,
        terminate_on_tolerance=args.terminate_on_tolerance,
    )


def obs_to_tensors(obs, device, batch=True):
    trace, loc, caoi = obs
    trace = torch.as_tensor(trace, dtype=torch.float32, device=device)
    loc = torch.as_tensor(loc, dtype=torch.float32, device=device)
    caoi = torch.as_tensor(caoi, dtype=torch.float32, device=device)
    if batch:
        trace = trace.unsqueeze(0)
        loc = loc.unsqueeze(0)
        caoi = caoi.unsqueeze(0)
    return trace, loc, caoi


def observe_current(transition, encoder, prev_state, prev_belief, prev_action, obs):
    trace, loc, caoi = obs_to_tensors(obs, prev_state.device, batch=True)
    embedding = encoder(trace, loc, caoi)
    out = transition(
        prev_state,
        prev_action.unsqueeze(0),
        prev_belief,
        observations=embedding.unsqueeze(0),
    )
    belief = out[0][0]
    posterior = out[4][0]
    return belief, posterior


def predict_current(transition, prev_state, prev_belief, prev_action):
    out = transition(prev_state, prev_action.unsqueeze(0), prev_belief, observations=None)
    return out[0][0], out[1][0]



def collect_seed_episode(env, replay):
    obs = env.reset()
    total_reward = 0.0
    for _ in range(env.max_episode_length):
        schedule = env.sample_random_action()
        action_logits = env.schedule_to_logits(schedule)
        next_obs, reward, done, _ = env.step(schedule)
        replay.append(obs, action_logits, reward, done)
        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


def run_policy_episode(
    args,
    env,
    transition,
    encoder,
    actor,
    device,
    replay=None,
    explore=False,
):
    obs = env.reset()
    belief = torch.zeros(1, args.belief_size, device=device)
    state = torch.zeros(1, args.state_size, device=device)
    prev_action = torch.zeros(1, actor.action_size, device=device)
    total_reward = 0.0
    final_caoi = float("nan")

    transition.eval(); encoder.eval(); actor.eval()
    with torch.no_grad():
        for t in range(env.max_episode_length):
            use_observation = (t == 0) or (random.random() >= args.prediction_drop_prob)
            if use_observation:
                belief, state = observe_current(
                    transition, encoder, state, belief, prev_action, obs
                )
            else:
                belief, state = predict_current(
                    transition, state, belief, prev_action
                )

            logits = actor(belief, state)
            if explore:
                logits = logits + args.action_noise * torch.randn_like(logits)
            schedule = actor.decode_schedule(logits[0])
            next_obs, reward, done, info = env.step(schedule)

            if replay is not None:
                replay.append(obs, logits[0].cpu(), reward, done)
            total_reward += reward
            final_caoi = info["average_caoi"]
            prev_action = logits.detach()
            obs = next_obs
            if done:
                break

    transition.train(); encoder.train(); actor.train()
    return total_reward, final_caoi


def save_checkpoint(path, episode, models, optimizers, args):
    payload = {
        "episode": episode,
        "args": vars(args),
        **{name: model.state_dict() for name, model in models.items()},
        **{name: opt.state_dict() for name, opt in optimizers.items()},
    }
    torch.save(payload, path)


def main():
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    if not Path(args.scene).is_absolute():
        args.scene = str((base_dir / args.scene).resolve())

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device("cpu")
    args.device_resolved = str(device)

    results_dir = base_dir / "results" / str(args.run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    writer = SummaryWriter(str(results_dir))

    print("Device:", device)
    print("Scene:", args.scene)
    print("Carrier frequency: %.1f GHz" % (args.frequency / 1e9))

    env = make_env(args)
    action_size = env.action_size

    transition = TransitionModel(
        args.belief_size, args.state_size, action_size,
        args.hidden_size, args.embedding_size
    ).to(device)
    encoder = MultimodalEncoder(
        args.num_v, env.trace_feature_dim, args.embedding_size
    ).to(device)
    decoder = MultimodalDecoder(
        args.num_v, env.trace_feature_dim,
        args.belief_size, args.state_size
    ).to(device)
    reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size).to(device)
    actor = ActorModel(args.belief_size, args.state_size, args.hidden_size, args.num_v).to(device)
    value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size).to(device)
    logic_model = LogicIntegratedNetwork(
        state_dim=args.belief_size + args.state_size,
        action_dim=action_size,
        logic_dim=args.logic_vector_size,
        reasoning_depth=args.reasoning_depth,
    ).to(device)

    system1_params = (
        list(transition.parameters()) + list(encoder.parameters())
        + list(decoder.parameters()) + list(reward_model.parameters())
    )
    model_optimizer = optim.Adam(system1_params, lr=args.model_learning_rate, eps=args.adam_epsilon)
    logic_optimizer = optim.SGD(logic_model.parameters(), lr=args.logic_learning_rate)
    model_logic_optimizer = optim.Adam(
        transition.parameters(), lr=args.model_logic_learning_rate, eps=args.adam_epsilon
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_learning_rate, eps=args.adam_epsilon)
    value_optimizer = optim.Adam(value_model.parameters(), lr=args.value_learning_rate, eps=args.adam_epsilon)

    models = {
        "transition": transition,
        "encoder": encoder,
        "decoder": decoder,
        "reward_model": reward_model,
        "actor": actor,
        "value_model": value_model,
        "logic_model": logic_model,
    }
    optimizers = {
        "model_optimizer": model_optimizer,
        "logic_optimizer": logic_optimizer,
        "model_logic_optimizer": model_logic_optimizer,
        "actor_optimizer": actor_optimizer,
        "value_optimizer": value_optimizer,
    }

    start_episode = 1
    if args.models:
        checkpoint = torch.load(args.models, map_location=device)
        for name, model in models.items():
            if name in checkpoint:
                model.load_state_dict(checkpoint[name])
        for name, opt in optimizers.items():
            if name in checkpoint:
                opt.load_state_dict(checkpoint[name])
        start_episode = int(checkpoint.get("episode", 0)) + 1
        print("Loaded checkpoint:", args.models)

    if args.test:
        rewards, cao_is = [], []
        for _ in tqdm(range(args.test_episodes), desc="test"):
            reward, caoi = run_policy_episode(
                args, env, transition, encoder, actor, device, replay=None, explore=False
            )
            rewards.append(reward); cao_is.append(caoi)
        print(f"Average test reward: {np.mean(rewards):.4f}")
        print(f"Average final CAoI: {np.mean(cao_is):.4f}")
        return

    replay = ExperienceReplay(
        args.experience_size, args.num_v, env.trace_feature_dim, action_size, device
    )
    print("Collecting seed episodes...")
    for s in range(args.seed_episodes):
        r = collect_seed_episode(env, replay)
        print(f"  seed {s+1}/{args.seed_episodes}: reward={r:.3f}, replay={len(replay)}")

    if len(replay) < args.sequence_length:
        raise RuntimeError(
            "Seed data are shorter than the requested sequence length. Increase "
            "seed episodes / max episode length or disable tolerance termination."
        )

    metrics = []
    for episode in range(start_episode, args.episodes + 1):
        update_stats = []
        for _ in tqdm(range(args.collect_interval), desc=f"train {episode}", leave=False):
            batch = replay.sample(args.batch_size, args.sequence_length)
            stats = train_one_update(
                args, batch, transition, encoder, decoder, reward_model,
                actor, value_model, logic_model,
                model_optimizer, logic_optimizer, model_logic_optimizer,
                actor_optimizer, value_optimizer,
            )
            update_stats.append(stats)

        train_reward, train_caoi = run_policy_episode(
            args, env, transition, encoder, actor, device, replay=replay, explore=True
        )
        mean_stats = {
            key: float(np.mean([s[key] for s in update_stats]))
            for key in update_stats[0]
        }
        mean_stats.update({
            "episode": episode,
            "steps": replay.steps,
            "train_reward": train_reward,
            "train_final_caoi": train_caoi,
        })
        metrics.append(mean_stats)

        for key, value in mean_stats.items():
            if key not in ("episode", "steps"):
                writer.add_scalar(key, value, replay.steps)

        print(
            f"episode={episode:04d} steps={replay.steps} "
            f"reward={train_reward:.3f} CAoI={train_caoi:.3f} "
            f"S1={mean_stats['model_loss']:.4f} "
            f"S2={mean_stats['logic_loss']:.4f} "
            f"logic->S1={mean_stats['logic_guidance_loss']:.4f}"
        )

        if episode % args.test_interval == 0:
            test_rewards, test_caoi = [], []
            for _ in range(args.test_episodes):
                r, c = run_policy_episode(
                    args, env, transition, encoder, actor, device, replay=None, explore=False
                )
                test_rewards.append(r); test_caoi.append(c)
            writer.add_scalar("test_reward", float(np.mean(test_rewards)), replay.steps)
            writer.add_scalar("test_final_caoi", float(np.mean(test_caoi)), replay.steps)
            print(
                f"  test reward={np.mean(test_rewards):.3f}, "
                f"final CAoI={np.mean(test_caoi):.3f}"
            )

        with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if episode % args.checkpoint_interval == 0:
            save_checkpoint(
                results_dir / f"models_{episode}.pth", episode, models, optimizers, args
            )
            if args.checkpoint_experience:
                torch.save(replay, results_dir / "experience.pth")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
