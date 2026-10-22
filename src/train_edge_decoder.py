from __future__ import annotations

import argparse
import copy
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from edge_decoder import (
    SymmetricEdgeDecoder,
    append_horizon_feature,
    temporal_pair_features,
)


def load_edges(directory: Path) -> dict[int, frozenset[tuple[int, int]]]:
    snapshots = {}
    for path in sorted(directory.glob("*.txt"), key=lambda value: int(value.stem)):
        edges = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split(chr(35), 1)[0].split()
            if len(fields) < 2:
                continue
            u, v = int(fields[0]), int(fields[1])
            if u != v:
                edges.add((u, v) if u < v else (v, u))
        snapshots[int(path.stem)] = frozenset(edges)
    return snapshots


def sample_case(
    item, target_edges, history, rng, max_positives, negative_ratio,
    *, horizon, max_horizon,
):
    embeddings = torch.as_tensor(item["embedding"], dtype=torch.float32)
    indices = np.asarray(item.get("indices"), dtype=np.int64)
    if indices.ndim != 1 or len(indices) != embeddings.shape[0]:
        indices = np.arange(embeddings.shape[0], dtype=np.int64)
    local = {int(node): position for position, node in enumerate(indices)}
    positives = [edge for edge in target_edges if edge[0] in local and edge[1] in local]
    if len(positives) > max_positives:
        chosen = rng.choice(len(positives), max_positives, replace=False)
        positives = [positives[int(index)] for index in chosen]
    if not positives:
        return None

    target = set(target_edges)
    historical = set().union(*history) if history else set()
    hard = [edge for edge in historical - target if edge[0] in local and edge[1] in local]
    needed = len(positives) * negative_ratio
    negatives = []
    hard_needed = min(len(hard), needed // 2)
    if hard_needed:
        chosen = rng.choice(len(hard), hard_needed, replace=False)
        negatives.extend(hard[int(index)] for index in chosen)
    negative_set = set(negatives)
    while len(negatives) < needed:
        selected = rng.choice(indices, size=(max(needed - len(negatives), 256), 2))
        for u, v in selected:
            if u == v:
                continue
            edge = (int(u), int(v)) if u < v else (int(v), int(u))
            if edge in target or edge in negative_set:
                continue
            negative_set.add(edge)
            negatives.append(edge)
            if len(negatives) >= needed:
                break
    pairs = positives + negatives
    pair_indices = torch.tensor(
        [(local[u], local[v]) for u, v in pairs], dtype=torch.long
    )
    history_features = torch.from_numpy(append_horizon_feature(
        temporal_pair_features(pairs, history), horizon, max_horizon
    ))
    labels = torch.cat(
        [torch.ones(len(positives)), torch.zeros(len(negatives))]
    )
    return embeddings, pair_indices, history_features, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-positives", type=int, default=2000)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--horizons", default="1")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "datasets" / args.dataset
    golden_path = root / "saved_models" / f"{args.dataset}_phase2_embedding_splits_GOLDEN.pkl"
    output_path = root / "saved_models" / f"edge_decoder_{args.dataset}.pth"
    with golden_path.open("rb") as handle:
        golden = pickle.load(handle)
    snapshots = load_edges(data_dir)
    train_items = sorted(golden["train"], key=lambda item: int(item["timestamp"]))
    validation_items = sorted(golden["val"], key=lambda item: int(item["timestamp"]))
    if not train_items or not validation_items:
        raise RuntimeError("Edge decoder requires non-empty train and validation embeddings")
    horizons = tuple(sorted(set(
        int(value) for value in args.horizons.split(",") if value
    )))
    if not horizons or horizons[0] < 1:
        raise ValueError("--horizons must contain positive integers")
    max_horizon = max(horizons)
    embedding_dim = int(train_items[0]["embedding"].shape[1])
    model = SymmetricEdgeDecoder(embedding_dim, history_dim=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(42)
    best_ap = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    all_items = sorted(train_items + validation_items, key=lambda item: int(item["timestamp"]))
    item_position = {int(item["timestamp"]): index for index, item in enumerate(all_items)}

    def build_case(item, horizon, negative_ratio):
        timestamp = int(item["timestamp"])
        position = item_position[timestamp]
        anchor_position = position - int(horizon)
        if anchor_position < 0:
            return None
        anchor_timestamp = int(all_items[anchor_position]["timestamp"])
        history = [
            snapshots[t] for t in sorted(snapshots)
            if t <= anchor_timestamp
        ]
        return sample_case(
            item,
            snapshots[timestamp],
            history,
            rng,
            args.max_positives,
            negative_ratio,
            horizon=horizon,
            max_horizon=max_horizon,
        )

    print("Preparing fixed causal train/validation edge samples...")
    train_cases = [
        case
        for item in train_items
        for horizon in horizons
        if (case := build_case(item, horizon, args.negative_ratio)) is not None
    ]
    validation_cases = [
        case
        for item in validation_items[-3:]
        for horizon in horizons
        if (case := build_case(item, horizon, 10)) is not None
    ]
    if not train_cases or not validation_cases:
        raise RuntimeError("No usable train/validation cases for the edge decoder")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for case in train_cases:
            embeddings, pairs, history_features, labels = case
            embeddings = embeddings.to(device)
            pairs = pairs.to(device)
            history_features = history_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                embeddings[pairs[:, 0]],
                embeddings[pairs[:, 1]],
                history_features,
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        labels_all, scores_all = [], []
        with torch.no_grad():
            for case in validation_cases:
                embeddings, pairs, history_features, labels = case
                logits = model(
                    embeddings.to(device)[pairs[:, 0].to(device)],
                    embeddings.to(device)[pairs[:, 1].to(device)],
                    history_features.to(device),
                )
                labels_all.extend(labels.numpy().tolist())
                scores_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        if not labels_all:
            raise RuntimeError("No validation cases for the edge decoder")
        validation_ap = float(average_precision_score(labels_all, scores_all))
        validation_auc = float(roc_auc_score(labels_all, scores_all))
        print(
            f"Edge decoder epoch {epoch + 1:02d} | "
            f"train={float(np.mean(train_losses)):.6f} | "
            f"val_auc={validation_auc:.6f} | val_ap={validation_ap:.6f}"
        )
        if validation_ap > best_ap + 1e-6:
            best_ap = validation_ap
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    torch.save(
        {
            "state_dict": best_state,
            "embedding_dim": embedding_dim,
            "history_dim": 6,
            "forecast_horizons": list(horizons),
            "max_horizon": max_horizon,
            "best_validation_ap": best_ap,
            "train_timestamps": [int(item["timestamp"]) for item in train_items],
            "validation_timestamps": [int(item["timestamp"]) for item in validation_items],
        },
        output_path,
    )
    print(output_path)


if __name__ == "__main__":
    main()
