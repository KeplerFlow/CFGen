from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Edge = tuple[int, int]


def canonical_edge(u: int, v: int) -> Edge | None:
    u, v = int(u), int(v)
    if u == v:
        return None
    return (u, v) if u < v else (v, u)


def temporal_pair_features(
    pairs: Sequence[Edge],
    history_edge_sets: Sequence[Iterable[Edge]],
    *,
    decay: float = 0.8,
    feature_index: dict[Edge, np.ndarray] | None = None,
) -> np.ndarray:
    if feature_index is not None:
        output = np.zeros((len(pairs), 5), dtype=np.float32)
        for index, edge in enumerate(pairs):
            value = feature_index.get(canonical_edge(*edge))
            if value is not None:
                output[index] = value
        return output
    allowed = {canonical_edge(*edge): index for index, edge in enumerate(pairs)}
    count = len(pairs)
    last = np.zeros(count, dtype=np.float32)
    count5 = np.zeros(count, dtype=np.float32)
    count20 = np.zeros(count, dtype=np.float32)
    exponential = np.zeros(count, dtype=np.float32)
    recency = np.zeros(count, dtype=np.float32)
    for age, edges in enumerate(reversed(history_edge_sets), 1):
        weight = decay ** (age - 1)
        for u, v in edges:
            index = allowed.get(canonical_edge(u, v))
            if index is None:
                continue
            if age == 1:
                last[index] = 1.0
            if age <= 5:
                count5[index] += 0.2
            if age <= 20:
                count20[index] += 0.05
            exponential[index] += weight
            if recency[index] == 0.0:
                recency[index] = 1.0 / age
    normalizer = sum(decay ** age for age in range(len(history_edge_sets)))
    if normalizer > 0:
        exponential /= normalizer
    return np.stack([last, count5, count20, exponential, recency], axis=1)


def append_horizon_feature(
    history_features: np.ndarray | torch.Tensor,
    horizon: int,
    max_horizon: int,
) -> np.ndarray | torch.Tensor:
    horizon = int(horizon)
    max_horizon = int(max_horizon)
    if horizon < 1 or max_horizon < horizon:
        raise ValueError("Expected 1 <= horizon <= max_horizon")
    value = float(np.log1p(horizon) / np.log1p(max_horizon))
    if isinstance(history_features, torch.Tensor):
        column = torch.full(
            (history_features.shape[0], 1), value,
            dtype=history_features.dtype, device=history_features.device,
        )
        return torch.cat((history_features, column), dim=1)
    array = np.asarray(history_features)
    column = np.full((array.shape[0], 1), value, dtype=array.dtype)
    return np.concatenate((array, column), axis=1)


def build_temporal_edge_feature_index(
    history_edge_sets: Sequence[Iterable[Edge]],
    *,
    decay: float = 0.8,
) -> dict[Edge, np.ndarray]:
    normalizer = sum(decay ** age for age in range(len(history_edge_sets))) or 1.0
    output: dict[Edge, np.ndarray] = {}
    for age, edges in enumerate(reversed(history_edge_sets), 1):
        weight = decay ** (age - 1) / normalizer
        for u, v in edges:
            edge = canonical_edge(u, v)
            if edge is None:
                continue
            values = output.setdefault(edge, np.zeros(5, dtype=np.float32))
            if age == 1:
                values[0] = 1.0
            if age <= 5:
                values[1] += 0.2
            if age <= 20:
                values[2] += 0.05
            values[3] += weight
            if values[4] == 0.0:
                values[4] = 1.0 / age
    return output


class SymmetricEdgeDecoder(nn.Module):

    def __init__(self, embedding_dim: int, history_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.history_dim = int(history_dim)
        input_dim = 2 * self.embedding_dim + 2 + self.history_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        history_features: torch.Tensor,
    ) -> torch.Tensor:
        product = left * right
        absolute = torch.abs(left - right)
        cosine = F.cosine_similarity(left, right).unsqueeze(1)
        distance = torch.linalg.vector_norm(left - right, dim=1, keepdim=True)
        features = torch.cat(
            [product, absolute, cosine, distance, history_features], dim=1
        )
        return self.network(features).squeeze(1)


def score_edge_pairs(
    model: SymmetricEdgeDecoder,
    embeddings: torch.Tensor,
    pair_indices: torch.Tensor,
    history_features: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    device = next(model.parameters()).device
    pair_indices = pair_indices.long().cpu()
    history = torch.as_tensor(history_features, dtype=torch.float32)
    values = []
    model.eval()
    with torch.no_grad():
        local_embeddings = embeddings.float().to(device)
        for start in range(0, pair_indices.shape[0], batch_size):
            stop = min(start + batch_size, pair_indices.shape[0])
            batch_pairs = pair_indices[start:stop].to(device)
            logits = model(
                local_embeddings[batch_pairs[:, 0]],
                local_embeddings[batch_pairs[:, 1]],
                history[start:stop].to(device),
            )
            values.append(torch.sigmoid(logits).cpu().numpy())
    if not values:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(values).astype(np.float32, copy=False)
