from __future__ import annotations

import copy
import gc
from typing import Dict, Generator, List

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix, negative_sampling
from tqdm import tqdm

from light_model import SpatioTemporalGCN


class TemporalEmbedder:
    def __init__(self, gnn_model: SpatioTemporalGCN, temporal_decay: float = 0.8):
        if not 0.0 <= temporal_decay < 1.0:
            raise ValueError("temporal_decay must be in [0, 1)")
        self.gnn_model = gnn_model
        self.temporal_decay = float(temporal_decay)
        self.gnn_model.eval()

    @staticmethod
    def _as_dense_tensor(features) -> torch.Tensor:
        if sp.issparse(features):
            features = features.toarray()
        return torch.from_numpy(np.asarray(features, dtype=np.float32))

    @staticmethod
    def _temporal_view(ema: torch.Tensor, temporal_dim: int) -> torch.Tensor:
        if ema.shape[1] == temporal_dim:
            return ema
        if ema.shape[1] > temporal_dim:
            chunks = torch.tensor_split(ema, temporal_dim, dim=1)
            return torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)
        return F.pad(ema, (0, temporal_dim - ema.shape[1]))

    def _causal_temporal_states(self, aligned_gnn_inputs: List[Dict]) -> List[torch.Tensor]:
        states: List[torch.Tensor] = []
        ema = None
        temporal_dim = int(self.gnn_model.lin_temporal.in_features)
        for snapshot in aligned_gnn_inputs:
            current = self._as_dense_tensor(snapshot["features"])
            if ema is None:
                ema = current.clone()
            else:
                ema.mul_(self.temporal_decay).add_(current, alpha=1.0 - self.temporal_decay)
            states.append(self._temporal_view(ema, temporal_dim).contiguous())
        return states

    @staticmethod
    def _sample_edges(
        edge_index: torch.Tensor,
        max_edges: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_index
        positive = edge_index[:, edge_index[0] < edge_index[1]]
        if positive.shape[1] > max_edges:
            order = torch.randperm(positive.shape[1], generator=generator)[:max_edges]
            positive = positive[:, order]
        return positive

    def fit_link_prediction(
        self,
        train_inputs: List[Dict],
        validation_inputs: List[Dict],
        *,
        epochs: int = 8,
        snapshots_per_epoch: int = 12,
        max_edges_per_snapshot: int = 8000,
        learning_rate: float = 1e-3,
        patience: int = 3,
        seed: int = 42,
    ) -> float:
        if not train_inputs:
            raise ValueError("Phase-2 link training requires training snapshots")
        device = next(self.gnn_model.parameters()).device
        sequence = list(train_inputs) + list(validation_inputs)
        temporal_states = self._causal_temporal_states(sequence)
        train_count = len(train_inputs)
        train_candidates = list(range(1, train_count)) or [0]
        if len(train_candidates) > snapshots_per_epoch:
            selected = np.linspace(
                0, len(train_candidates) - 1, snapshots_per_epoch, dtype=int
            )
            train_indices = [train_candidates[int(index)] for index in selected]
        else:
            train_indices = train_candidates
        validation_indices = list(range(train_count, len(sequence)))[-3:]
        if not validation_indices:
            validation_indices = train_indices[-1:]

        optimizer = torch.optim.AdamW(
            self.gnn_model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        generator = torch.Generator().manual_seed(seed)
        best_loss = float("inf")
        best_state = copy.deepcopy(self.gnn_model.state_dict())
        stale = 0

        def snapshot_loss(index: int):
            snapshot = sequence[index]
            spatial = self._as_dense_tensor(snapshot["features"]).to(device)
            temporal = temporal_states[index].to(device)
            edge_index, _ = from_scipy_sparse_matrix(snapshot["adj"])
            positive = self._sample_edges(
                edge_index, max_edges_per_snapshot, generator
            )
            if positive.numel() == 0:
                return None
            positive = positive.to(device)
            edge_index = edge_index.to(device)
            negative = negative_sampling(
                edge_index=edge_index,
                num_nodes=spatial.shape[0],
                num_neg_samples=positive.shape[1],
                method="sparse",
            )
            embedding = self.gnn_model(spatial, temporal, edge_index)
            scale = max(1.0, float(embedding.shape[1]) ** 0.5)
            positive_logits = (
                embedding[positive[0]] * embedding[positive[1]]
            ).sum(1) / scale
            negative_logits = (
                embedding[negative[0]] * embedding[negative[1]]
            ).sum(1) / scale
            logits = torch.cat([positive_logits, negative_logits])
            labels = torch.cat(
                [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)]
            )
            return F.binary_cross_entropy_with_logits(logits, labels)

        for epoch in range(max(1, int(epochs))):
            self.gnn_model.train()
            train_losses = []
            for index in train_indices:
                optimizer.zero_grad(set_to_none=True)
                loss = snapshot_loss(index)
                if loss is None:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gnn_model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            self.gnn_model.eval()
            validation_losses = []
            with torch.no_grad():
                for index in validation_indices:
                    loss = snapshot_loss(index)
                    if loss is not None:
                        validation_losses.append(float(loss.detach().cpu()))
            if not train_losses:
                raise RuntimeError("Phase-2 link training found no positive edges")
            validation_loss = float(np.mean(validation_losses or train_losses))
            print(
                f"Phase2 link epoch {epoch + 1:02d} | "
                f"train={float(np.mean(train_losses)):.6f} | "
                f"val={validation_loss:.6f}"
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.gnn_model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        self.gnn_model.load_state_dict(best_state)
        self.gnn_model.eval()
        return best_loss

    def generate_embedding_sequence(
        self,
        aligned_gnn_inputs: List[Dict],
        batch_size: int = 50000,
    ) -> Generator[Dict, None, None]:
        del batch_size
        if not aligned_gnn_inputs:
            return
        device = next(self.gnn_model.parameters()).device
        temporal_states = self._causal_temporal_states(aligned_gnn_inputs)
        print(
            "Generating causal spatiotemporal embeddings "
            f"(EMA decay={self.temporal_decay:.3f})..."
        )
        self.gnn_model.eval()
        with torch.no_grad():
            for index, snapshot in enumerate(
                tqdm(aligned_gnn_inputs, desc="Generating spatiotemporal embeddings")
            ):
                spatial = self._as_dense_tensor(snapshot["features"]).to(device)
                temporal = temporal_states[index].to(device)
                edge_index, _ = from_scipy_sparse_matrix(snapshot["adj"])
                edge_index = edge_index.to(device)
                embedding = self.gnn_model(spatial, temporal, edge_index)
                yield {
                    "timestamp": snapshot["timestamp"],
                    "embedding": embedding.cpu(),
                }
                del spatial, temporal, edge_index, embedding
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del temporal_states
        gc.collect()
