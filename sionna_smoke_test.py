"""Run a small Sionna RT sanity check in the target Sionna environment.

This checks the specific issues that previously failed in the notebook:
LLVM configuration, 26-GHz carrier setting, Mitsuba scalar conversion,
dynamic vehicle motion/re-tracing, CIR shape, self-link masking, channel gain,
and ray-tracing-based packets/timeslot.
"""
from __future__ import annotations

import numpy as np

from runtime import configure_drjit_llvm

# Must run before importing env -> Mitsuba/Sionna.
llvm = configure_drjit_llvm()
if llvm:
    print("DRJIT_LIBLLVM_PATH:", llvm)

from env import SionnaV2XEnv  # noqa: E402


def _scalar(x):
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy()).item()
    return np.asarray(x).item()


def main():
    env = SionnaV2XEnv(
        num_v=3,
        num_antennas=4,
        max_episode_length=3,
        # Smaller smoke-test ray budget; use main.py defaults for experiments.
        max_num_paths_per_src=1000,
        samples_per_src=10000,
        diffuse_reflection=False,
        terminate_on_tolerance=False,
    )
    try:
        obs = env.reset()
        fc = _scalar(env.scene.frequency)
        assert abs(fc - 26e9) < 1e3, fc
        print(f"carrier_frequency={fc/1e9:.1f} GHz")
        print("trace shape:", obs[0].shape)
        print("location shape:", obs[1].shape)
        print("CAoI shape:", obs[2].shape)
        print("gain shape:", env.last_rt.channel_gain.shape)
        print("rate shape:", env.last_rt.packets_per_slot.shape)

        assert obs[0].shape == env.trace_shape
        assert obs[1].shape == env.location_shape
        assert obs[2].shape == env.caoi_shape
        assert np.all(np.isfinite(env.last_rt.channel_gain))
        assert np.all(np.isfinite(env.last_rt.packets_per_slot))
        assert np.all(env.last_rt.channel_gain >= 0.0)

        # Vehicle TX i+1 -> RX i is a self-link and must be masked.
        for i in range(env.num_v):
            assert env.last_rt.channel_gain[i + 1, i] == 0.0

        before = env.raw_locations.copy()
        schedule = env.sample_random_action()
        _next_obs, reward, done, info = env.step(schedule)
        after = env.raw_locations.copy()
        moved = np.mod(after[:, 0] - before[:, 0], env.road_length)
        expected = env.speeds * env.slot_duration
        assert np.allclose(moved, expected, atol=1e-5), (moved, expected)

        assert np.isfinite(reward)
        assert np.all(np.isfinite(info["packets_per_slot"]))
        print("schedule:", info["schedule"])
        print("movement [m]:", moved)
        print("average CAoI:", info["average_caoi"])
        print("V2I rates [packets/slot]:", info["packets_per_slot"][0])
        print("SIONNA_SMOKE_TEST_OK")
    finally:
        env.close()


if __name__ == "__main__":
    main()
