import torch
import numpy as np
import pickle
import os
import networkx as nx
from collections import defaultdict
from torch_geometric.data import Data
import logging
from typing import List, Dict, Tuple, Set

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import scipy.sparse as sp
from coarse_graph import CoarseGraph

def run_coarsener_validation(
    coarse_graph: CoarseGraph,
    hidden_edge_index: torch.Tensor,
    all_visible_edges_adj: sp.csr_matrix,
    embeddings_to_validate: np.ndarray,
    num_nodes: int
) -> float:
    if hidden_edge_index.shape[1] == 0:
        return 0.0

    node_to_supernode_map = coarse_graph.node_to_supernode
    supernode_features = coarse_graph.features
    uplifted_embeddings = torch.tensor(supernode_features[node_to_supernode_map])

    pos_edges = hidden_edge_index.cpu().detach()
    num_neg_samples = pos_edges.size(1)

    neg_edges = negative_sampling(
        edge_index=torch.tensor(all_visible_edges_adj.nonzero()),
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method='sparse'
    )

    total_edges = torch.cat([pos_edges, neg_edges], dim=-1)
    labels = torch.cat([
        torch.ones(pos_edges.size(1)),
        torch.zeros(neg_edges.size(1))
    ], dim=0)

    scores = (uplifted_embeddings[total_edges[0]] * uplifted_embeddings[total_edges[1]]).sum(dim=-1)

    try:
        auc = roc_auc_score(labels.numpy(), scores.numpy())
    except ValueError as e:
        auc = 0.5

    return auc

def _center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - (1/n) * np.ones((n, n))
    return H @ K @ H

def cka(X: np.ndarray, Y: np.ndarray) -> float:
    K_X = linear_kernel(X)
    K_Y = linear_kernel(Y)

    K_X_c = _center_kernel(K_X)
    K_Y_c = _center_kernel(K_Y)

    hsic = (K_X_c * K_Y_c).sum()
    norm_X = np.sqrt((K_X_c * K_X_c).sum())
    norm_Y = np.sqrt((K_Y_c * K_Y_c).sum())

    if norm_X * norm_Y == 0:
        return 0.0
    return hsic / (norm_X * norm_Y)

def run_cka_evaluation_on_split(
    golden_sequence: List[torch.Tensor],
    coarse_sequence: List[torch.Tensor]
) -> float:
    all_ckas = []
    desc = "Eval CKA (structural)"

    for i in tqdm(range(len(golden_sequence)), desc=desc, leave=False):
        emb_golden = golden_sequence[i].cpu().detach().numpy()
        emb_coarse = coarse_sequence[i].cpu().detach().numpy()

        if emb_golden.shape[0] != emb_coarse.shape[0]:
            logging.warning(f"Snapshot {i}: embedding shape mismatch, skipping CKA.")
            continue

        if emb_golden.shape[0] == 0:
            continue

        cka_score = cka(emb_golden, emb_coarse)
        all_ckas.append(cka_score)

    return np.mean(all_ckas) if all_ckas else 0.0

def find_community_knn(
    embeddings: np.ndarray,
    query_node: int,
    k: int
) -> Set[int]:
    if query_node >= embeddings.shape[0]:
        return {query_node}

    query_vec = embeddings[query_node].reshape(1, -1)
    sims = cosine_similarity(embeddings, query_vec).flatten()
    top_k_indices = np.argsort(sims)[-k:]
    return set(top_k_indices)

def evaluate_community_quality(
    found_community: Set[int],
    predicted_subgraph: nx.Graph,
    time_window: int = 1
) -> Dict[str, float]:
    S = found_community
    G = predicted_subgraph

    metrics = {}

    if not S or len(S) < 2:
        return {
            "Community Nodes": len(S), "Community Edges": 0,
            "Density": 0.0, "TC": 0.0, "TD": 0.0
        }

    metrics["Community Nodes"] = len(S)
    community_subgraph = G.subgraph(S)
    metrics["Community Edges"] = community_subgraph.number_of_edges()

    E_S_internal = metrics["Community Edges"]
    denominator_td = len(S) * (len(S) - 1) * time_window
    metrics["TD"] = (2 * E_S_internal) / denominator_td if denominator_td > 0 else 0.0

    try:
        Tcut_S = nx.cut_size(G, S)

        if Tcut_S == 0:
            metrics["TC"] = 0.0
        else:
            Tvol_S = nx.volume(G, S)
            total_volume = sum(d for n, d in G.degree())
            Tvol_S_bar = total_volume - Tvol_S
            denominator_tc = min(Tvol_S, Tvol_S_bar)

            if denominator_tc > 0:
                metrics["TC"] = Tcut_S / denominator_tc
            else:
                metrics["TC"] = float('inf')

    except Exception:
        metrics["TC"] = float('inf')

    metrics["Density"] = nx.density(community_subgraph)

    return metrics

def run_community_search_evaluation_on_split(
    embedding_sequence: List[torch.Tensor],
    all_edges_list: List[torch.Tensor],
    num_nodes: int,
    num_query_nodes: int = 100,
    community_size_k: int = 20
) -> Dict[str, float]:
    all_metrics = defaultdict(list)

    desc = "Eval community search (functional)"
    for i in tqdm(range(len(embedding_sequence)), desc=desc, leave=False):
        emb = embedding_sequence[i].cpu().detach().numpy()

        edges = all_edges_list[i].cpu().detach().T.numpy()
        if edges.shape[0] == 0:
            continue

        G_truth = nx.Graph()
        G_truth.add_nodes_from(range(num_nodes))
        G_truth.add_edges_from(edges)

        if num_query_nodes >= num_nodes:
            query_nodes = list(range(num_nodes))
        else:
            query_nodes = np.random.choice(num_nodes, num_query_nodes, replace=False)

        snapshot_metrics = defaultdict(list)

        for q_node in query_nodes:
            community = find_community_knn(emb, q_node, community_size_k)
            quality = evaluate_community_quality(community, G_truth, time_window=1)

            for key, val in quality.items():
                snapshot_metrics[key].append(val)

        for key, val_list in snapshot_metrics.items():
            if val_list:
                all_metrics[key].append(np.mean(val_list))

    final_avg_metrics = {}
    for key, val_list in all_metrics.items():
        if val_list:
            final_avg_metrics[key] = np.mean(val_list)

    return final_avg_metrics
