"""Experience replay for multimodal V2X observations."""
from __future__ import annotations

import numpy as np
import torch


class ExperienceReplay:
    def __init__(
        self,
        size: int,
        num_v: int,
        trace_feature_dim: int,
        action_size: int,
        device,
    ):
        self.size = int(size)
        self.num_v = int(num_v)
        self.trace_feature_dim = int(trace_feature_dim)
        self.action_size = int(action_size)
        self.device = device

        self.traces = np.empty(
            (self.size, num_v, num_v + 1, trace_feature_dim), dtype=np.float32
        )
        self.locations = np.empty((self.size, num_v, 3), dtype=np.float32)
        self.caoi = np.empty((self.size, num_v), dtype=np.float32)
        self.actions = np.empty((self.size, action_size), dtype=np.float32)
        self.rewards = np.empty((self.size,), dtype=np.float32)
        self.nonterminals = np.empty((self.size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0

    def __len__(self):
        return self.size if self.full else self.idx

    def append(self, observation, action, reward: float, done: bool):
        trace, loc, caoi = observation
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        self.traces[self.idx] = np.asarray(trace, dtype=np.float32)
        self.locations[self.idx] = np.asarray(loc, dtype=np.float32)
        self.caoi[self.idx] = np.asarray(caoi, dtype=np.float32)
        self.actions[self.idx] = np.asarray(action, dtype=np.float32)
        self.rewards[self.idx] = float(reward)
        self.nonterminals[self.idx, 0] = 0.0 if done else 1.0

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        self.episodes += int(done)

    def _candidate_indices(self, length: int) -> np.ndarray:
        available = len(self)
        if available < length:
            raise RuntimeError(
                f"Replay buffer contains only {available} transitions, but sequence length is {length}."
            )

        # Rejection sampling keeps chunks inside one real episode. This avoids
        # silently joining two reset states into one RSSM training trajectory.
        for _ in range(10000):
            if self.full:
                start = np.random.randint(0, self.size)
                idxs = (start + np.arange(length)) % self.size
                # Do not cross the current write pointer.
                if self.idx in idxs[1:]:
                    continue
            else:
                start = np.random.randint(0, available - length + 1)
                idxs = np.arange(start, start + length)

            # Terminal is allowed only at the final stored transition.
            if np.any(self.nonterminals[idxs[:-1], 0] == 0.0):
                continue
            return idxs
        raise RuntimeError("Could not sample an episode-consistent replay chunk.")

    def sample(self, batch_size: int, sequence_length: int):
        idxs = np.stack(
            [self._candidate_indices(sequence_length) for _ in range(batch_size)], axis=0
        )  # [B,L]
        flat = idxs.T.reshape(-1)
        L, B = sequence_length, batch_size

        def tm(array):
            x = torch.from_numpy(array[flat].astype(np.float32))
            x = x.reshape(L, B, *x.shape[1:])
            return x.to(self.device)

        return (
            tm(self.traces),
            tm(self.locations),
            tm(self.caoi),
            tm(self.actions),
            tm(self.rewards),
            tm(self.nonterminals),
        )
