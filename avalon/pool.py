"""Opponent pool: frozen past copies of the network (PLAN.md section 5)."""

import copy
from collections import deque


class OpponentPool:
    def __init__(self, every=10, size=20):
        self.every = every
        self.snapshots = deque(maxlen=size)

    def maybe_add(self, update, model):
        if update % self.every == 0:
            snap = copy.deepcopy(model).eval()
            for p in snap.parameters():
                p.requires_grad_(False)
            self.snapshots.append(snap)

    def sample(self, rng):
        if not self.snapshots:
            return None
        return self.snapshots[int(rng.integers(len(self.snapshots)))]

    def __len__(self):
        return len(self.snapshots)
