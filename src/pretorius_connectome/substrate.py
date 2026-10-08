"""Small deterministic sparse recurrent substrate for Pretorius-Connectome.

This is a synthetic smoke-test graph, not imported biological connectivity.
No external dependencies are required.
"""
from dataclasses import dataclass
from math import tanh
from typing import Sequence


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    weight: float


class SparseRecurrentNetwork:
    """Synchronous tanh updates over directed weighted edges."""

    def __init__(self, size: int, edges: Sequence[Edge], leak: float = 0.5):
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0.0 <= leak <= 1.0:
            raise ValueError("leak must be in [0, 1]")
        self.size = size
        self.leak = leak
        self.edges = tuple(edges)
        for edge in self.edges:
            if not (0 <= edge.source < size and 0 <= edge.target < size):
                raise ValueError("edge endpoint outside network")
        self.state = [0.0] * size

    def step(self, stimulus: Sequence[float] | None = None) -> tuple[float, ...]:
        if stimulus is None:
            stimulus = [0.0] * self.size
        if len(stimulus) != self.size:
            raise ValueError("stimulus length does not match network size")
        current = self.state
        drive = [float(x) for x in stimulus]
        for edge in self.edges:
            drive[edge.target] += current[edge.source] * edge.weight
        self.state = [
            (1.0 - self.leak) * old + self.leak * tanh(total)
            for old, total in zip(current, drive)
        ]
        return tuple(self.state)

    def reset(self) -> None:
        self.state = [0.0] * self.size


if __name__ == "__main__":
    net = SparseRecurrentNetwork(
        4, [Edge(0, 1, 0.8), Edge(1, 2, 0.6), Edge(2, 3, -0.4)]
    )
    for index in range(5):
        state = net.step([1.0, 0.0, 0.0, 0.0] if index == 0 else None)
        print(f"tick {index + 1}: {state}")
