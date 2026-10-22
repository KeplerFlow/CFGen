from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np


Edge = tuple[int, int]


def canonical_edge(u: int, v: int) -> Edge | None:
    u, v = int(u), int(v)
    if u == v:
        return None
    return (u, v) if u < v else (v, u)


def _adjacency(edges: Iterable[Edge]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for raw_u, raw_v in edges:
        edge = canonical_edge(raw_u, raw_v)
        if edge is None:
            continue
        u, v = edge
        adjacency[u].add(v)
        adjacency[v].add(u)
    return dict(adjacency)


def _intersection_size(left: set[int], right: set[int]) -> int:
    if len(left) > len(right):
        left, right = right, left
    return sum(node in right for node in left)


class CausalPairFeatureBuilder:

    feature_names = (
        "edge_last",
        "edge_count_2",
        "edge_count_3",
        "edge_count_5",
        "edge_count_10",
        "edge_count_20",
        "edge_count_all",
        "edge_decay_050",
        "edge_decay_080",
        "edge_decay_095",
        "edge_inverse_recency",
        "edge_streak",
        "latest_degree_min_log",
        "latest_degree_max_log",
        "latest_degree_product_log",
        "mean5_degree_min_log",
        "mean5_degree_max_log",
        "degree_trend_sum",
        "degree_trend_absdiff",
        "activity20_min",
        "activity20_max",
        "latest_common_log",
        "latest_jaccard",
        "recent5_common_log",
        "recent5_jaccard",
        "union_common_log",
        "union_jaccard",
    )

    def __init__(self, history_edge_sets: Sequence[Iterable[Edge]]):
        self.history = [
            frozenset(
                edge
                for raw in edges
                if (edge := canonical_edge(*raw)) is not None
            )
            for edges in history_edge_sets
        ]
        if not self.history:
            raise ValueError("Pair features require at least one history snapshot")

        self.edge_ages: dict[Edge, list[int]] = defaultdict(list)
        for age, edges in enumerate(reversed(self.history), 1):
            for edge in edges:
                self.edge_ages[edge].append(age)

        self.latest_adjacency = _adjacency(self.history[-1])
        recent_edges = set().union(*self.history[-5:])
        union_edges = set().union(*self.history)
        self.recent_adjacency = _adjacency(recent_edges)
        self.union_adjacency = _adjacency(union_edges)

        degree_sums: dict[int, float] = defaultdict(float)
        older_degree_sums: dict[int, float] = defaultdict(float)
        activity: dict[int, int] = defaultdict(int)
        recent_snapshots = self.history[-5:]
        older_snapshots = self.history[-10:-5]
        for edges in recent_snapshots:
            adjacency = _adjacency(edges)
            for node, neighbors in adjacency.items():
                degree_sums[node] += len(neighbors)
        for edges in older_snapshots:
            adjacency = _adjacency(edges)
            for node, neighbors in adjacency.items():
                older_degree_sums[node] += len(neighbors)
        for edges in self.history[-20:]:
            for node in _adjacency(edges):
                activity[node] += 1
        recent_denominator = float(len(recent_snapshots))
        older_denominator = float(max(1, len(older_snapshots)))
        activity_denominator = float(min(20, len(self.history)))
        nodes = set(degree_sums) | set(older_degree_sums) | set(activity)
        self.mean5_degree = {
            node: degree_sums[node] / recent_denominator for node in nodes
        }
        self.degree_trend = {
            node: self.mean5_degree.get(node, 0.0)
            - older_degree_sums[node] / older_denominator
            for node in nodes
        }
        self.activity20 = {
            node: activity[node] / activity_denominator for node in nodes
        }

    @staticmethod
    def _normalized_count(ages: Sequence[int], window: int, available: int) -> float:
        denominator = float(min(window, available))
        return sum(age <= window for age in ages) / denominator

    @staticmethod
    def _decay(ages: Sequence[int], decay: float, available: int) -> float:
        denominator = sum(decay**offset for offset in range(available))
        if denominator == 0.0:
            return 0.0
        return sum(decay ** (age - 1) for age in ages) / denominator

    @staticmethod
    def _structural(adjacency: dict[int, set[int]], u: int, v: int) -> tuple[float, float]:
        left = adjacency.get(u, set())
        right = adjacency.get(v, set())
        common = _intersection_size(left, right)
        union_size = len(left) + len(right) - common
        return np.log1p(common), (common / union_size if union_size else 0.0)

    def transform(self, pairs: Sequence[Edge]) -> np.ndarray:
        features = np.empty((len(pairs), len(self.feature_names)), dtype=np.float32)
        available = len(self.history)
        latest_degree = {node: len(neighbors) for node, neighbors in self.latest_adjacency.items()}
        for index, raw_edge in enumerate(pairs):
            edge = canonical_edge(*raw_edge)
            if edge is None:
                raise ValueError("Self-loops are not valid ranking candidates")
            u, v = edge
            ages = self.edge_ages.get(edge, ())
            recency = min(ages, default=available + 1)
            streak = 0
            age_set = set(ages)
            while streak + 1 in age_set:
                streak += 1

            du, dv = latest_degree.get(u, 0), latest_degree.get(v, 0)
            mu, mv = self.mean5_degree.get(u, 0.0), self.mean5_degree.get(v, 0.0)
            tu, tv = self.degree_trend.get(u, 0.0), self.degree_trend.get(v, 0.0)
            au, av = self.activity20.get(u, 0.0), self.activity20.get(v, 0.0)
            latest_common, latest_jaccard = self._structural(self.latest_adjacency, u, v)
            recent_common, recent_jaccard = self._structural(self.recent_adjacency, u, v)
            union_common, union_jaccard = self._structural(self.union_adjacency, u, v)
            features[index] = (
                float(1 in age_set),
                self._normalized_count(ages, 2, available),
                self._normalized_count(ages, 3, available),
                self._normalized_count(ages, 5, available),
                self._normalized_count(ages, 10, available),
                self._normalized_count(ages, 20, available),
                len(ages) / float(available),
                self._decay(ages, 0.50, available),
                self._decay(ages, 0.80, available),
                self._decay(ages, 0.95, available),
                1.0 / recency,
                streak / float(min(20, available)),
                np.log1p(min(du, dv)),
                np.log1p(max(du, dv)),
                np.log1p(du * dv),
                np.log1p(min(mu, mv)),
                np.log1p(max(mu, mv)),
                tu + tv,
                abs(tu - tv),
                min(au, av),
                max(au, av),
                latest_common,
                latest_jaccard,
                recent_common,
                recent_jaccard,
                union_common,
                union_jaccard,
            )
        return features

    def history_score(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features[:, self.feature_names.index("edge_decay_080")])


RANKER_FEATURE_NAMES = CausalPairFeatureBuilder.feature_names + ("forecast_horizon_log",)


def append_ranker_horizon(
    features: np.ndarray,
    horizon: int,
    max_horizon: int,
) -> np.ndarray:
    horizon, max_horizon = int(horizon), int(max_horizon)
    if horizon < 1 or max_horizon < horizon:
        raise ValueError("Expected 1 <= horizon <= max_horizon")
    value = float(np.log1p(horizon) / np.log1p(max_horizon))
    column = np.full((len(features), 1), value, dtype=features.dtype)
    return np.concatenate((features, column), axis=1)


def graph_preserving_score(
    graph_membership: np.ndarray,
    history_score: np.ndarray,
    learned_score: np.ndarray,
    history_weight: float,
) -> np.ndarray:
    membership = np.asarray(graph_membership, dtype=np.bool_)
    history = np.clip(np.asarray(history_score, dtype=np.float64), 0.0, 1.0)
    learned = np.clip(np.asarray(learned_score, dtype=np.float64), 0.0, 1.0)
    weight = float(history_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("history_weight must be in [0, 1]")
    within_group = weight * history + (1.0 - weight) * learned
    return np.where(membership, 0.5005 + 0.499 * within_group, 0.499 * within_group)
