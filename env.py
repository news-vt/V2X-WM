"""Dynamic 26-GHz Sionna RT mmWave V2X environment.

The environment provides the observation used by System 1 of the world model:

    o_t = {Xi_t, L_t, A_t}

where Xi_t contains strongest-path ray-tracing features, L_t contains vehicle
locations, and A_t contains completeness-aware packet-age averages (CAoI).

The implementation intentionally keeps the radio simulator separate from the
world-model training code.  Sionna RT is used only to generate physical paths
and channel coefficients; the paper-style rate equation converts the resulting
link gains to packets/timeslot.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from runtime import configure_drjit_llvm

# Must be called before the imports below on macOS CPU installations.
configure_drjit_llvm()

import mitsuba as mi  # noqa: E402
import sionna.rt  # noqa: E402
from sionna.rt import (  # noqa: E402
    PathSolver,
    PlanarArray,
    Receiver,
    SceneObject,
    Transmitter,
    load_scene,
)


BOLTZMANN = 1.380649e-23


@dataclass
class RayTraceResult:
    paths: object
    trace_features: np.ndarray      # [rx, tx, feature]
    channel_gain: np.ndarray        # [tx, rx], linear
    capacity_bps: np.ndarray        # [tx, rx]
    packets_per_slot: np.ndarray    # [tx, rx]
    link_valid: np.ndarray          # [tx, rx], bool


class SionnaV2XEnv:
    """Time-slotted V2X environment backed by Sionna RT.

    Action convention
    -----------------
    ``schedule[j]`` is the sender selected for receiver vehicle ``j``:

    * ``-1``: vehicle ``j`` is idle or acts as a sender;
    * ``0``: RSU -> vehicle ``j``;
    * ``i`` (1..V): vehicle ``i-1`` -> vehicle ``j``.

    The environment sanitizes schedules so that a vehicle cannot transmit and
    receive simultaneously, self-links are forbidden, and a vehicle transmitter
    is used at most once per timeslot. The RSU may serve multiple vehicles,
    consistent with the paper only constraining vehicle half-duplex operation.
    """

    TRACE_FEATURES = (
        "strongest_path_gain_norm",
        "delay_norm",
        "theta_t_norm",
        "phi_t_norm",
        "theta_r_norm",
        "phi_r_norm",
        "valid",
    )

    def __init__(
        self,
        scene_path: str = "itu_scene/itu_scene.xml",
        max_episode_length: int = 100,
        num_v: int = 8,
        num_antennas: int = 4,
        road_length: float = 200.0,
        min_gap: float = 20.0,
        tx_power_dbm: float = 23.0,
        bandwidth: float = 100e6,
        carrier_frequency: float = 26e9,
        slot_duration: float = 0.1,
        packet_size_bytes: float = 5e6,
        num_packets: int = 25,
        caoi_tolerance: float = 8.0,
        seed: int = 41,
        noise_figure_db: float = 0.0,
        temperature_k: float = 290.0,
        max_depth: int = 5,
        max_num_paths_per_src: int = 4000,
        samples_per_src: int = 20000,
        diffuse_reflection: bool = True,
        synthetic_array: bool = True,
        terminate_on_tolerance: bool = True,
    ):
        self.base_dir = Path(__file__).resolve().parent
        self.scene_path = self._resolve_asset(scene_path)
        self.vehicle_meshes = [
            self._resolve_asset("vehicle_small.ply"),
            self._resolve_asset("vehicle_big.ply"),
        ]

        self.max_episode_length = int(max_episode_length)
        self.num_v = int(num_v)
        self.num_antennas = int(num_antennas)
        self.road_length = float(road_length)
        self.min_gap = float(min_gap)
        self.tx_power_dbm = float(tx_power_dbm)
        self.bandwidth = float(bandwidth)
        self.carrier_frequency = float(carrier_frequency)
        self.slot_duration = float(slot_duration)
        self.packet_size_bytes = float(packet_size_bytes)
        self.packet_size_bits = 8.0 * self.packet_size_bytes
        self.num_packets = int(num_packets)
        self.caoi_tolerance = float(caoi_tolerance)
        self.seed = int(seed)
        self.noise_figure_db = float(noise_figure_db)
        self.temperature_k = float(temperature_k)
        self.max_depth = int(max_depth)
        self.max_num_paths_per_src = int(max_num_paths_per_src)
        self.samples_per_src = int(samples_per_src)
        self.diffuse_reflection = bool(diffuse_reflection)
        self.synthetic_array = bool(synthetic_array)
        self.terminate_on_tolerance = bool(terminate_on_tolerance)

        self.rng = np.random.default_rng(self.seed)
        self.solver = PathSolver()

        self.lane_y = np.asarray([6.0, 10.0, 14.0], dtype=np.float32)
        self.small_height = 1.7
        self.big_height = 5.0
        self.antenna_offset = 0.3
        self.rsu_position = np.asarray([self.road_length / 2.0, 0.0, 10.0], dtype=np.float32)

        self.trace_feature_dim = len(self.TRACE_FEATURES)
        # Actor action = V sender-nomination logits + V matching rows.
        # Each matching row has V+2 choices: IDLE, RSU, and V vehicles.
        self.match_choices = self.num_v + 2
        self.action_size = self.num_v + self.num_v * self.match_choices
        self.trace_shape = (self.num_v, self.num_v + 1, self.trace_feature_dim)
        self.location_shape = (self.num_v, 3)
        self.caoi_shape = (self.num_v,)

        self.scene = None
        self.Tx_bs = None
        self.Tx_v: list[object] = []
        self.Rx_v: list[object] = []
        self.vehicle_objects: list[object] = []
        self.last_rt: Optional[RayTraceResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self):
        self.t = 0
        self._init_vehicles()
        self._init_packet_ages()
        self._init_scene()
        self.last_rt = self._ray_trace()
        return self._observation(self.last_rt)

    def step(self, schedule):
        schedule = self._as_schedule(schedule)
        schedule = self._sanitize_schedule(schedule)

        # Rates correspond to the physical state at the beginning of this slot.
        if self.last_rt is None:
            self.last_rt = self._ray_trace()
        rt_used = self.last_rt

        self._advance_packet_ages(schedule, rt_used.packets_per_slot)
        reward = self._reward()

        self.t += 1
        tolerance_violation = bool(np.any(self.caoi > self.caoi_tolerance))
        done = self.t >= self.max_episode_length
        if self.terminate_on_tolerance:
            done = done or tolerance_violation

        # Exogenous mobility evolves after the current scheduling decision.
        self._update_positions()
        self._move_scene_objects()
        self.last_rt = self._ray_trace()
        obs = self._observation(self.last_rt)

        info = {
            "schedule": schedule.copy(),
            "caoi": self.caoi.copy(),
            "average_caoi": float(np.mean(self.caoi)),
            "tolerance_violation": tolerance_violation,
            # Physical links actually used for this scheduling decision.
            "channel_gain": rt_used.channel_gain.copy(),
            "capacity_bps": rt_used.capacity_bps.copy(),
            "packets_per_slot": rt_used.packets_per_slot.copy(),
            "link_valid": rt_used.link_valid.copy(),
            # Physical state corresponding to the returned next observation.
            "next_channel_gain": self.last_rt.channel_gain.copy(),
            "next_capacity_bps": self.last_rt.capacity_bps.copy(),
            "next_packets_per_slot": self.last_rt.packets_per_slot.copy(),
            "next_link_valid": self.last_rt.link_valid.copy(),
        }
        return obs, float(reward), bool(done), info

    def close(self):
        self.scene = None
        self.last_rt = None

    @property
    def caoi(self) -> np.ndarray:
        # Row 0 is the RSU. The model observes only vehicle CAoI.
        return self.packet_ages[1:].mean(axis=1).astype(np.float32)

    @property
    def raw_locations(self) -> np.ndarray:
        return self.locations.astype(np.float32, copy=True)

    def sample_random_action(self) -> torch.Tensor:
        """Sample a valid random schedule using the same action convention."""
        V = self.num_v
        schedule = np.full(V, -1, dtype=np.int64)

        # Randomly nominate a subset of vehicles as V2V senders.
        sender_mask = self.rng.random(V) < 0.35
        available_vehicle_senders = [i + 1 for i in np.flatnonzero(sender_mask)]

        for rx in range(V):
            if sender_mask[rx]:
                continue
            if self.rng.random() < 0.25:
                continue

            candidates = [0] + [s for s in available_vehicle_senders if s != rx + 1]
            if not candidates:
                continue
            sender = int(self.rng.choice(candidates))
            schedule[rx] = sender
            if sender > 0 and sender in available_vehicle_senders:
                available_vehicle_senders.remove(sender)

        return torch.from_numpy(schedule)

    def schedule_to_logits(self, schedule, strength: float = 4.0) -> torch.Tensor:
        """Encode a discrete schedule into the actor's continuous action vector."""
        schedule = self._sanitize_schedule(self._as_schedule(schedule))
        V, S = self.num_v, self.match_choices

        # A vehicle is a sender if it appears as a positive sender index.
        used_vehicle_senders = {int(s) for s in schedule if int(s) > 0}
        sender_logits = np.full(V, -strength, dtype=np.float32)
        for sender in used_vehicle_senders:
            sender_logits[sender - 1] = strength

        # Matching classes:
        #   0 = IDLE, 1 = RSU, 2..V+1 = vehicle sender 1..V.
        match_logits = np.full((V, S), -strength, dtype=np.float32)
        for rx, sender in enumerate(schedule):
            cls = 0 if int(sender) < 0 else int(sender) + 1
            match_logits[rx, cls] = strength

        return torch.from_numpy(np.concatenate([sender_logits, match_logits.reshape(-1)]))

    # Backwards-compatible name used by the original environment.
    invert_actions_to_logits = schedule_to_logits

    # ------------------------------------------------------------------
    # Scene and mobility
    # ------------------------------------------------------------------
    def _resolve_asset(self, path: str | Path) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = self.base_dir / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Required asset not found: {path}")
        return path

    @staticmethod
    def _point3(x, y, z):
        # Explicit float conversion avoids Mitsuba rejecting numpy.float64.
        return mi.Point3f([float(x), float(y), float(z)])

    @staticmethod
    def _vector3(x, y, z):
        return mi.Vector3f([float(x), float(y), float(z)])

    def _init_vehicles(self):
        locations: list[list[float]] = []
        speeds: list[float] = []
        types: list[int] = []
        lanes: list[int] = []

        lane_counts = self.rng.multinomial(self.num_v, np.ones(len(self.lane_y)) / len(self.lane_y))
        for lane_id, count in enumerate(lane_counts):
            if count == 0:
                continue
            xs = self._generate_collision_free_positions(int(count))

            # A common speed per lane preserves the configured safety distance
            # in this no-lane-change test environment.
            lane_speed = float(self.rng.uniform(15.0, 20.0))
            for x in xs:
                vehicle_type = 1 if lane_id == 0 else int(self.rng.choice([0, 1], p=[0.8, 0.2]))
                height = self.big_height if vehicle_type == 1 else self.small_height
                locations.append([float(x), float(self.lane_y[lane_id]), float(height)])
                speeds.append(lane_speed)
                types.append(vehicle_type)
                lanes.append(lane_id)

        # The multinomial loop should create exactly V vehicles.
        self.locations = np.asarray(locations[: self.num_v], dtype=np.float64)
        self.speeds = np.asarray(speeds[: self.num_v], dtype=np.float64)
        self.types = np.asarray(types[: self.num_v], dtype=np.int64)
        self.lanes = np.asarray(lanes[: self.num_v], dtype=np.int64)

    def _generate_collision_free_positions(self, count: int) -> np.ndarray:
        """Generate positions with the safety gap preserved across road wrap-around.

        Vehicles in the same lane use a common velocity. Treating the 200-m road
        as cyclic therefore preserves these gaps throughout an episode, including
        when a vehicle exits at x=L and re-enters at x=0.
        """
        if count <= 0:
            return np.empty((0,), dtype=np.float64)
        if count == 1:
            return np.asarray([self.rng.uniform(0.0, self.road_length)], dtype=np.float64)

        required = count * self.min_gap
        if required > self.road_length:
            raise ValueError(
                f"Cannot place {count} vehicles on a cyclic {self.road_length} m lane "
                f"with minimum gap {self.min_gap} m."
            )

        # There are `count` circular gaps. Give each gap the safety distance and
        # distribute the remaining road length randomly.
        remainder = self.road_length - required
        extra = self.rng.dirichlet(np.ones(count)) * remainder
        gaps = self.min_gap + extra

        offset = self.rng.uniform(0.0, self.road_length)
        positions = [offset]
        for gap in gaps[:-1]:
            positions.append(positions[-1] + gap)
        return np.sort(np.mod(np.asarray(positions), self.road_length))

    def _init_scene(self):
        self.scene = load_scene(str(self.scene_path), merge_shapes=False)

        # This is the actual RF carrier setting. Paths.cir(sampling_frequency=...)
        # controls time sampling, not the carrier frequency.
        self.scene.frequency = self.carrier_frequency
        # Keep scene metadata synchronized when the installed Sionna version
        # exposes these properties. The rate calculation below is still explicit.
        for attr, value in (("bandwidth", self.bandwidth), ("temperature", self.temperature_k)):
            try:
                setattr(self.scene, attr, value)
            except Exception:
                pass

        # Sionna RT uses one TX array for all transmitters and one RX array for
        # all receivers. The paper uses four antennas, so RSU and vehicle TXs
        # share the same 1xN array here.
        self.scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=self.num_antennas,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V",
        )
        self.scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=self.num_antennas,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V",
        )

        self.Tx_v = []
        self.Rx_v = []
        self.vehicle_objects = []

        try:
            self.Tx_bs = Transmitter(
                name="tx_bs",
                position=self._point3(*self.rsu_position),
                power_dbm=self.tx_power_dbm,
            )
        except TypeError:
            self.Tx_bs = Transmitter(name="tx_bs", position=self._point3(*self.rsu_position))
            if hasattr(self.Tx_bs, "power_dbm"):
                self.Tx_bs.power_dbm = self.tx_power_dbm
        self.scene.add(self.Tx_bs)

        vehicle_material = self._vehicle_material()
        for i in range(self.num_v):
            x, y, height = self.locations[i]
            antenna_z = height + self.antenna_offset
            pos = self._point3(x, y, antenna_z)
            velocity = self._vector3(self.speeds[i], 0.0, 0.0)

            try:
                tx = Transmitter(
                    name=f"tx_v_{i}", position=pos, velocity=velocity, power_dbm=self.tx_power_dbm
                )
            except TypeError:
                tx = Transmitter(name=f"tx_v_{i}", position=pos)
                tx.velocity = velocity
                if hasattr(tx, "power_dbm"):
                    tx.power_dbm = self.tx_power_dbm

            try:
                rx = Receiver(name=f"rx_v_{i}", position=pos, velocity=velocity)
            except TypeError:
                rx = Receiver(name=f"rx_v_{i}", position=pos)
                rx.velocity = velocity

            self.scene.add(tx)
            self.scene.add(rx)
            self.Tx_v.append(tx)
            self.Rx_v.append(rx)

            obj = SceneObject(
                name=f"vehicle_{i}",
                fname=str(self.vehicle_meshes[int(self.types[i])]),
                radio_material=vehicle_material,
            )
            self.scene.edit(add=obj)
            obj.position = self._point3(x, y, height / 2.0)
            try:
                obj.velocity = velocity
            except Exception:
                pass
            self.vehicle_objects.append(obj)

        self._geometry_updated()

    def _vehicle_material(self):
        for name in ("itu_metal", "metal", "mat-itu_metal"):
            try:
                material = self.scene.get(name)
                if material is not None:
                    return material
            except Exception:
                continue
        raise RuntimeError(
            "No metal radio material was found in the loaded scene. Expected "
            "one of: itu_metal, metal, mat-itu_metal."
        )

    def _geometry_updated(self):
        callback = getattr(self.scene, "scene_geometry_updated", None)
        if callable(callback):
            callback()

    def _update_positions(self):
        self.locations[:, 0] = np.mod(
            self.locations[:, 0] + self.speeds * self.slot_duration,
            self.road_length,
        )

    def _move_scene_objects(self):
        for i, (x, y, height) in enumerate(self.locations):
            antenna_z = height + self.antenna_offset
            self.vehicle_objects[i].position = self._point3(x, y, height / 2.0)
            self.Tx_v[i].position = self._point3(x, y, antenna_z)
            self.Rx_v[i].position = self._point3(x, y, antenna_z)
            velocity = self._vector3(self.speeds[i], 0.0, 0.0)
            try:
                self.vehicle_objects[i].velocity = velocity
                self.Tx_v[i].velocity = velocity
                self.Rx_v[i].velocity = velocity
            except Exception:
                pass
        self._geometry_updated()

    # ------------------------------------------------------------------
    # Ray tracing and physical link rate
    # ------------------------------------------------------------------
    def _solver_kwargs(self) -> dict:
        kwargs = dict(
            scene=self.scene,
            max_depth=self.max_depth,
            max_num_paths_per_src=self.max_num_paths_per_src,
            samples_per_src=self.samples_per_src,
            los=True,
            specular_reflection=True,
            diffuse_reflection=self.diffuse_reflection,
            refraction=True,
            synthetic_array=self.synthetic_array,
            seed=self.seed + self.t,
        )
        # Filter arguments for compatibility with nearby Sionna RT releases.
        try:
            sig = inspect.signature(self.solver.__call__)
            return {k: v for k, v in kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            return kwargs

    def _ray_trace(self) -> RayTraceResult:
        paths = self.solver(**self._solver_kwargs())
        a, _tau = paths.cir(
            num_time_steps=1,
            normalize_delays=False,
            reverse_direction=False,
            out_type="numpy",
        )
        a = np.asarray(a)
        if a.ndim != 6:
            raise RuntimeError(
                "Unexpected Sionna CIR shape. Expected "
                "[rx, rx_ant, tx, tx_ant, path, time], got "
                f"{a.shape}."
            )

        gain = self._effective_channel_gain(a)
        trace_features = self._strongest_path_features(paths, a)
        snr, capacity_bps, packets_per_slot = self._rate_from_gain(gain)
        link_valid = gain > 0.0
        return RayTraceResult(
            paths=paths,
            trace_features=trace_features,
            channel_gain=gain,
            capacity_bps=capacity_bps,
            packets_per_slot=packets_per_slot,
            link_valid=link_valid,
        )

    def _effective_channel_gain(self, a: np.ndarray) -> np.ndarray:
        """Return directional rank-1 gain sigma_max(H)^2 for each link.

        The CIR coefficients are coherently summed over multipath components to
        form a narrowband MIMO channel matrix. The dominant singular value gives
        a reproducible scalar directional-link abstraction. The paper specifies
        narrow directional beams but does not specify a beamformer, so this is
        an explicit implementation choice rather than a claimed paper formula.
        """
        # Remove time and coherently combine paths.
        h = np.sum(a[..., 0], axis=-1)  # [rx, rx_ant, tx, tx_ant]
        num_rx, _, num_tx, _ = h.shape
        gain = np.zeros((num_tx, num_rx), dtype=np.float64)
        for tx in range(num_tx):
            for rx in range(num_rx):
                H = h[rx, :, tx, :]
                if not np.any(np.abs(H) > 0.0):
                    continue
                try:
                    sigma = np.linalg.svd(H, compute_uv=False)
                    gain[tx, rx] = float(np.abs(sigma[0]) ** 2)
                except np.linalg.LinAlgError:
                    gain[tx, rx] = float(np.linalg.norm(H, ord="fro") ** 2)

        # TX 0 is the RSU; TX i+1 and RX i belong to the same vehicle.
        for i in range(min(self.num_v, num_rx)):
            if i + 1 < num_tx:
                gain[i + 1, i] = 0.0
        gain[~np.isfinite(gain)] = 0.0
        return gain

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy())
        return np.asarray(value)

    def _path_value(self, paths, name: str, rx: int, tx: int, p: int) -> float:
        arr = self._to_numpy(getattr(paths, name))
        try:
            if arr.ndim == 3:  # synthetic arrays: [rx, tx, path]
                value = arr[rx, tx, p]
            elif arr.ndim == 5:  # explicit arrays: [rx, rx_ant, tx, tx_ant, path]
                values = arr[rx, :, tx, :, p].reshape(-1)
                values = values[np.isfinite(values)]
                value = values[0] if values.size else 0.0
            else:
                return 0.0
            value = float(value)
            return value if math.isfinite(value) else 0.0
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _strongest_path_features(self, paths, a: np.ndarray) -> np.ndarray:
        # Per-path received energy aggregated over antenna pairs.
        path_power = np.sum(np.abs(a[..., 0]) ** 2, axis=(1, 3))  # [rx, tx, path]
        num_rx, num_tx, num_paths = path_power.shape
        features = np.zeros((num_rx, num_tx, self.trace_feature_dim), dtype=np.float32)

        for rx in range(num_rx):
            for tx in range(num_tx):
                powers = path_power[rx, tx]
                if num_paths == 0 or not np.any(powers > 0.0):
                    continue
                p = int(np.argmax(powers))
                pwr = float(powers[p])
                gain_db = 10.0 * np.log10(max(pwr, 1e-30))
                delay = max(0.0, self._path_value(paths, "tau", rx, tx, p))
                theta_t = self._path_value(paths, "theta_t", rx, tx, p)
                phi_t = self._path_value(paths, "phi_t", rx, tx, p)
                theta_r = self._path_value(paths, "theta_r", rx, tx, p)
                phi_r = self._path_value(paths, "phi_r", rx, tx, p)

                # Normalize to stable ranges for neural-network training.
                features[rx, tx] = np.asarray(
                    [
                        np.clip((gain_db + 180.0) / 180.0, 0.0, 1.0),
                        np.clip(delay / 5e-6, 0.0, 2.0),
                        np.clip(theta_t / np.pi, -1.0, 1.0),
                        np.clip(phi_t / np.pi, -1.0, 1.0),
                        np.clip(theta_r / np.pi, -1.0, 1.0),
                        np.clip(phi_r / np.pi, -1.0, 1.0),
                        1.0,
                    ],
                    dtype=np.float32,
                )

        # Mask co-located vehicle self-links in the observation too.
        for i in range(min(self.num_v, num_rx)):
            if i + 1 < num_tx:
                features[i, i + 1] = 0.0
        return features

    def _rate_from_gain(self, gain: np.ndarray):
        p_tx_w = 10.0 ** ((self.tx_power_dbm - 30.0) / 10.0)
        n0 = BOLTZMANN * self.temperature_k * (10.0 ** (self.noise_figure_db / 10.0))
        noise_power = n0 * self.bandwidth
        snr = p_tx_w * gain / max(noise_power, 1e-30)
        snr = np.maximum(snr, 0.0)
        capacity_bps = self.bandwidth * np.log2(1.0 + snr)
        packets_per_slot = capacity_bps * self.slot_duration / self.packet_size_bits
        return snr, capacity_bps, packets_per_slot

    # ------------------------------------------------------------------
    # Completeness-aware packet age / CAoI
    # ------------------------------------------------------------------
    def _init_packet_ages(self):
        # One row for RSU + one row per vehicle. The RSU creates a fresh set of
        # Cu packets each slot; vehicles start with slightly stale information.
        self.packet_ages = np.ones((self.num_v + 1, self.num_packets), dtype=np.float32)
        self.packet_ages[1:] = self.rng.integers(
            2, 4, size=(self.num_v, self.num_packets)
        ).astype(np.float32)

    def _advance_packet_ages(self, schedule: np.ndarray, packets_per_slot: np.ndarray):
        # Time advances for information stored at vehicles.
        self.packet_ages[1:] += 1.0
        # RSU has a newly generated Cu-packet update every slot.
        self.packet_ages[0].fill(1.0)

        for rx_vehicle, sender in enumerate(schedule):
            if sender < 0:
                continue
            receiver_row = rx_vehicle + 1
            sender_row = int(sender)  # 0=RSU, i=vehicle i-1

            if sender_row > self.num_v or sender_row == receiver_row:
                continue
            if sender_row >= packets_per_slot.shape[0] or rx_vehicle >= packets_per_slot.shape[1]:
                continue

            capacity = int(np.floor(max(0.0, packets_per_slot[sender_row, rx_vehicle])))
            if capacity <= 0:
                continue

            source = self.packet_ages[sender_row]
            receiver = self.packet_ages[receiver_row]
            fresher = np.flatnonzero(source < receiver)
            if fresher.size == 0:
                continue

            # Select the packets that produce the largest freshness improvement.
            improvement = receiver[fresher] - source[fresher]
            order = fresher[np.argsort(-improvement)]
            selected = order[: min(capacity, fresher.size, self.num_packets)]
            receiver[selected] = source[selected]

    def _reward(self) -> float:
        # Paper reward: -1/V sum_v [A_v - I(A_v>Abar)(Abar-A_v)].
        age = self.caoi.astype(np.float64)
        penalized = age.copy()
        over = age > self.caoi_tolerance
        penalized[over] = age[over] - (self.caoi_tolerance - age[over])
        return -float(np.mean(penalized))

    # ------------------------------------------------------------------
    # Observation/action helpers
    # ------------------------------------------------------------------
    def _observation(self, rt: RayTraceResult):
        locations = self.locations.astype(np.float32).copy()
        locations[:, 0] /= max(self.road_length, 1.0)
        locations[:, 1] /= max(float(np.max(self.lane_y)), 1.0)
        locations[:, 2] /= max(self.big_height, 1.0)

        caoi = self.caoi / max(self.caoi_tolerance, 1.0)
        caoi = np.clip(caoi, 0.0, 4.0).astype(np.float32)
        return (
            rt.trace_features.astype(np.float32, copy=True),
            locations,
            caoi,
        )

    def _as_schedule(self, schedule) -> np.ndarray:
        if isinstance(schedule, torch.Tensor):
            schedule = schedule.detach().cpu().numpy()
        schedule = np.asarray(schedule, dtype=np.int64).reshape(-1)
        if schedule.size != self.num_v:
            raise ValueError(f"Expected schedule with {self.num_v} entries, got {schedule.shape}")
        return schedule

    def _sanitize_schedule(self, schedule: np.ndarray) -> np.ndarray:
        schedule = schedule.copy()
        schedule[(schedule < -1) | (schedule > self.num_v)] = -1

        # Discover vehicle senders and then enforce half-duplex/unique-use.
        used_vehicle_senders: set[int] = set()
        for rx in range(self.num_v):
            sender = int(schedule[rx])
            if sender <= 0:
                continue
            if sender == rx + 1 or sender in used_vehicle_senders:
                schedule[rx] = -1
                continue
            used_vehicle_senders.add(sender)

        # A vehicle used as a sender cannot simultaneously be a receiver.
        for sender in used_vehicle_senders:
            schedule[sender - 1] = -1
        return schedule


# Backwards-compatible alias used by the earlier project.
Sionna_mmWave_V2X_Env = SionnaV2XEnv
