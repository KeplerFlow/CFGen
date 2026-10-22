import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
import networkx as nx
import pandas as pd
from typing import List, Set, Dict, Union, Any
import time
import scipy.sparse as sp
import joblib
import argparse
import contextlib
import gc
import random
import psutil
import shutil
import json
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import f1_score, recall_score, jaccard_score, normalized_mutual_info_score
from torch_geometric.utils import degree

try:
    from torch.amp import autocast
    def auto_cast_ctx():
        return autocast('cuda')
except ImportError:
    try:
        from torch.cuda.amp import autocast
        def auto_cast_ctx():
            return autocast()
    except ImportError:
        @contextlib.contextmanager
        def auto_cast_ctx():
            yield

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from model import BlockDenoisingModelV14
from diffusion import Diffusion
from main import load_temporal_graph
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph, to_undirected
from train_refinement import EmbeddingRefinementGNN_V15_1

class DiskBasedDataManager:
    def __init__(self, original_pkl_path: str, cache_dir_base: str = None):
        self.original_path = original_pkl_path
        if cache_dir_base is None:
            cache_dir_base = os.path.join(PROJECT_ROOT, "cache")
        filename = os.path.basename(original_pkl_path).replace('.', '_')
        self.cache_dir = os.path.join(cache_dir_base, filename)
        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
        self.metadata = []
        self._length = 0

        if not self._is_cache_valid():
            self._build_cache()
        else:
            self._load_metadata()

    def _is_cache_valid(self):
        if not os.path.exists(self.metadata_file):
            return False
        try:
            with open(self.metadata_file, 'r') as f:
                meta = json.load(f)
                if not meta: return False
                last_idx = len(meta) - 1
                return os.path.exists(os.path.join(self.cache_dir, f"frame_{last_idx}.npy"))
        except:
            return False

    def _build_cache(self):
        print(f"\n[DiskManager] Initializing cache from {self.original_path}...")
        print(f"[DiskManager] This minimizes RAM usage for future runs. Please wait...")
        print(f"[DiskManager] Cache directory: {self.cache_dir}")

        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self._print_mem("Before Big Load")

        try:
            data = joblib.load(self.original_path, mmap_mode='r')
        except:
            with open(self.original_path, 'rb') as f:
                data = pickle.load(f)

        self._print_mem("After Big Load")

        all_items = []
        if isinstance(data, dict):
            all_items.extend(data.get('train', []))
            all_items.extend(data.get('val', []))
            all_items.extend(data.get('test', []))
        elif isinstance(data, list):
            all_items = data
        else:
            raise ValueError("Unknown data format in pickle")

        self.metadata = []

        print(f"[DiskManager] converting {len(all_items)} snapshots to disk...")

        for idx, item in enumerate(all_items):

            emb = item.get('embedding', None)

            meta_entry = {
                'idx': idx,
                'timestamp': item.get('timestamp', idx),
                'is_sparse': sp.issparse(emb),
                'shape': emb.shape if hasattr(emb, 'shape') else (len(emb), len(emb[0])) if emb else (0,0)
            }

            file_path = os.path.join(self.cache_dir, f"frame_{idx}")

            if sp.issparse(emb):
                sp.save_npz(file_path, emb)
                meta_entry['file_type'] = 'npz'
            elif isinstance(emb, (np.ndarray, np.memmap)):
                np.save(file_path, emb)
                meta_entry['file_type'] = 'npy'
            elif isinstance(emb, list):
                arr = np.array(emb, dtype=np.float32)
                np.save(file_path, arr)
                meta_entry['file_type'] = 'npy'
            else:
                meta_entry['file_type'] = 'none'

            self.metadata.append(meta_entry)

            if idx % 100 == 0:
                print(f"\r[DiskManager] Processed {idx}/{len(all_items)}", end="")

        print("\n[DiskManager] Saving metadata...")
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

        self._length = len(self.metadata)

        del data
        del all_items
        gc.collect()
        self._print_mem("After Cache Build & GC")

    def _load_metadata(self):
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self._length = len(self.metadata)
        print(f"[DiskManager] Cache loaded. {self._length} snapshots available via disk.")

    def _print_mem(self, tag):
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024 / 1024
        print(f"[Memory {tag}] RSS: {mem:.2f} GB")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._length:
            raise IndexError("Index out of bounds")

        meta = self.metadata[idx]
        file_base = os.path.join(self.cache_dir, f"frame_{idx}")

        res = {
            'timestamp': meta['timestamp'],
            'embedding': None
        }

        if meta['file_type'] == 'npy':
            res['embedding'] = np.load(file_base + ".npy", mmap_mode='r')
        elif meta['file_type'] == 'npz':
            res['embedding'] = sp.load_npz(file_base + ".npz")

        return res

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"[Memory {tag}] Current RSS: {mem:.2f} GB")

def extract_local_embedding(raw_item, subset_indices, device, num_nodes_global):
    target_nodes_cpu = subset_indices.cpu().numpy()
    embedding_data = raw_item.get('embedding', None)

    if embedding_data is None:
        return torch.zeros((len(target_nodes_cpu), 64), device=device)

    local_emb = None

    if isinstance(embedding_data, (np.ndarray, np.memmap)):
        try:
            slice_data = embedding_data[target_nodes_cpu]
            local_emb = torch.from_numpy(slice_data).float()
        except IndexError:
            dim = embedding_data.shape[1]
            local_emb = torch.zeros((len(target_nodes_cpu), dim), dtype=torch.float32)
            valid_mask = target_nodes_cpu < embedding_data.shape[0]
            if valid_mask.any():
                valid_indices = target_nodes_cpu[valid_mask]
                data_slice = embedding_data[valid_indices]
                local_emb[valid_mask] = torch.from_numpy(data_slice).float()

    elif sp.issparse(embedding_data):
        try:
            slice_sparse = embedding_data[target_nodes_cpu]
            local_emb = torch.from_numpy(slice_sparse.toarray()).float()
        except:
             local_emb = torch.zeros((len(target_nodes_cpu), 64), device=device)

    if local_emb is None:
        local_emb = torch.zeros((len(target_nodes_cpu), 64), dtype=torch.float32)

    return local_emb.to(device)

def perform_local_search_ssge_p(subgraph: nx.Graph, subgraph_embedding: torch.Tensor, query_seeds, alpha: float = 0.7) -> Set[int]:
    if isinstance(query_seeds, int): seeds = {query_seeds}
    else: seeds = set(query_seeds)

    if subgraph.number_of_nodes() == 0: return seeds
    valid_seeds = {s for s in seeds if s in subgraph.nodes()}
    if not valid_seeds: return seeds

    subgraph_nodes = sorted(list(subgraph.nodes()))
    node_to_idx = {n: i for i, n in enumerate(subgraph_nodes)}

    seed_indices = [node_to_idx[s] for s in valid_seeds]
    seed_indices_tensor = torch.tensor(seed_indices, device=subgraph_embedding.device)

    q_emb = subgraph_embedding.index_select(0, seed_indices_tensor).mean(dim=0).unsqueeze(0)

    sims = F.cosine_similarity(q_emb, subgraph_embedding).detach().cpu().numpy()
    sims = (sims + 1.0) / 2.0

    seed_scores = sims[seed_indices]
    avg_seed_score = np.mean(seed_scores)
    threshold = alpha * avg_seed_score

    community = set(valid_seeds)
    visited = set(valid_seeds)
    queue = list(valid_seeds)

    while queue:
        curr_node = queue.pop(0)

        for neighbor in subgraph.neighbors(curr_node):
            if neighbor not in visited:
                visited.add(neighbor)

                if neighbor in node_to_idx:
                    nbr_idx = node_to_idx[neighbor]
                    score = sims[nbr_idx]

                    if score >= threshold:
                        community.add(neighbor)
                        queue.append(neighbor)

    relaxed_threshold = threshold * 0.90
    for node in subgraph_nodes:
        if node not in community:
            idx = node_to_idx[node]
            if sims[idx] >= relaxed_threshold:
                community.add(node)

    return community

SDR_ITERS = 1
SDR_ANNEAL = 0.7
SDR_SOFT_BETA = 4.0
SDR_SOFT_TAU = 0.30
SDR_COH_SCALE = 4.0

EDS_TOP_M = 5000
EDS_TELEPORT = 0.15
EDS_ITERS = 60
EDS_LOOKBACK = 3
EDS_DECAY = 0.85

def _sem_score(sem_sum_ns, n_nonseed, alpha):
    return (sem_sum_ns / (n_nonseed ** alpha)) if n_nonseed > 0 else 0.0

def _sege_energy(sem_score, n_nodes, n_int_edges, vol, cut, total_graph_vol, lambda_bal, size_weight=0.3):
    td = (2.0 * n_int_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 2 else 0.0
    if cut <= 0:
        tc = 0.0
    else:
        vol_bar = total_graph_vol - vol
        denom_tc = min(vol, vol_bar)
        tc = cut / denom_tc if denom_tc > 0 else 1.0
    import math
    size_term = math.log(n_nodes + 1)
    return (1.0 - lambda_bal) * sem_score + lambda_bal * (td - tc) + size_weight * size_term

def sege_energy_search(subgraph: nx.Graph, subgraph_embedding: torch.Tensor,
                       query_seeds, min_community_size: int = 3,
                       max_community_size: int = 200, lambda_bal: float = 0.5,
                       alpha: float = 0.7, query_embedding: torch.Tensor = None,
                       sim_sharp: float = 1.0, size_weight: float = 0.3, degree_weight: float = 0.5) -> Set[int]:
    seeds = {query_seeds} if isinstance(query_seeds, int) else set(query_seeds)
    if subgraph.number_of_nodes() == 0:
        return seeds
    valid_seeds = {s for s in seeds if s in subgraph.nodes()}
    if not valid_seeds:
        return seeds

    subgraph_nodes = sorted(list(subgraph.nodes()))
    node_to_idx = {n: i for i, n in enumerate(subgraph_nodes)}
    if subgraph_embedding.shape[0] != len(subgraph_nodes):
        neighbors = set()
        for s in valid_seeds:
            neighbors.update(subgraph.neighbors(s))
        return neighbors | valid_seeds

    dev = subgraph_embedding.device
    if query_embedding is None:
        seed_idx = torch.tensor([node_to_idx[s] for s in valid_seeds], device=dev)
        q_emb = subgraph_embedding.index_select(0, seed_idx).mean(dim=0).unsqueeze(0)
    else:
        q_emb = query_embedding.reshape(1, -1).to(dev)

    sims = F.cosine_similarity(q_emb, subgraph_embedding).cpu().numpy()
    sims = ((sims + 1.0) / 2.0) * float(sim_sharp)
    sim_map = {n: float(sims[node_to_idx[n]]) for n in subgraph_nodes}
    is_seed = {n: (n in valid_seeds) for n in subgraph_nodes}

    total_graph_vol = 2 * subgraph.number_of_edges()
    EPSILON = 1e-6

    adj = {n: set(subgraph.neighbors(n)) for n in subgraph_nodes}
    deg = {n: len(adj[n]) for n in subgraph_nodes}

    def sem_of(ns_sum, n_ns):
        return _sem_score(ns_sum, n_ns, alpha)

    import numpy as _np
    import math as _math
    N = len(subgraph_nodes)
    nbrs = [[] for _ in range(N)]
    for n in subgraph_nodes:
        ni = node_to_idx[n]
        for m in adj[n]:
            nbrs[ni].append(node_to_idx[m])
    deg_np = _np.array([deg[n] for n in subgraph_nodes], dtype=_np.int64)
    sim_np = _np.array([sim_map[n] for n in subgraph_nodes], dtype=_np.float64)
    is_seed_np = _np.array([1 if is_seed[n] else 0 for n in subgraph_nodes], dtype=_np.int64)
    in_S = _np.zeros(N, dtype=bool)
    edges_to_S = _np.zeros(N, dtype=_np.int64)
    seed_local = [node_to_idx[s] for s in valid_seeds]
    in_S[seed_local] = True
    n_nodes = len(seed_local)
    vol = int(deg_np[seed_local].sum())
    n_int_edges = 0
    for si in seed_local:
        for m in nbrs[si]:
            if in_S[m]:
                n_int_edges += 1
    n_int_edges //= 2
    for si in seed_local:
        for m in nbrs[si]:
            if not in_S[m]:
                edges_to_S[m] += 1
    sem_sum_ns = 0.0
    n_nonseed = 0
    cut = vol - 2 * n_int_edges
    current_energy = _sege_energy(sem_of(sem_sum_ns, n_nonseed), n_nodes, n_int_edges,
                                  vol, cut, total_graph_vol, lambda_bal, size_weight)
    max_energy = current_energy
    best_S_mask = in_S.copy()
    patience_counter = 0
    while n_nodes < max_community_size:
        bnd = _np.where((edges_to_S > 0) & (~in_S))[0]
        if bnd.size == 0:
            break
        d_int = edges_to_S[bnd]
        d_vol = deg_np[bnd]
        nonseed_b = (1 - is_seed_np[bnd])
        d_sem = sim_np[bnd] * nonseed_b
        d_ns = nonseed_b
        new_n = n_nodes + 1
        new_int = n_int_edges + d_int
        new_vol = vol + d_vol
        new_cut = cut + d_vol - 2 * d_int
        new_ns = n_nonseed + d_ns
        td = _np.where(new_n > 2, (2.0 * new_int) / (new_n * (new_n - 1)), 0.0)
        vol_bar = total_graph_vol - new_vol
        denom = _np.minimum(new_vol, vol_bar)
        tc = _np.where((new_cut > 0) & (denom > 0), new_cut / _np.maximum(denom, 1), 0.0)
        sem_sc = _np.where(new_ns > 0, (sem_sum_ns + d_sem) / (new_ns ** alpha), 0.0)
        new_e = (1 - lambda_bal) * sem_sc + lambda_bal * (td - tc) + size_weight * _math.log(new_n + 1)
        gain = new_e - current_energy + degree_weight * (deg_np[bnd].astype(_np.float64) / max(1.0, float(deg_np.max())))
        bi = int(_np.argmax(gain))
        best_gain = float(gain[bi])
        bv = int(bnd[bi])
        patience_counter = patience_counter + 1 if best_gain <= EPSILON else 0
        if patience_counter > 5:
            break
        in_S[bv] = True
        n_nodes += 1
        n_int_edges += int(d_int[bi])
        vol += int(d_vol[bi])
        cut += int(d_vol[bi]) - 2 * int(d_int[bi])
        if not is_seed_np[bv]:
            sem_sum_ns += float(sim_np[bv])
            n_nonseed += 1
        current_energy = float(new_e[bi])
        edges_to_S[bv] = 0
        for m in nbrs[bv]:
            if not in_S[m]:
                edges_to_S[m] += 1
        if current_energy > max_energy + EPSILON:
            max_energy = current_energy
            best_S_mask = in_S.copy()
    best_community = {subgraph_nodes[i] for i in _np.where(best_S_mask)[0]}

    if len(best_community) <= 1:
        return best_community | valid_seeds

    community = best_community
    edges_inside = {v: 0 for v in community}
    for v in community:
        for u in adj[v]:
            if u in community:
                edges_inside[v] += 1
    n_nodes = len(community)
    n_int_edges = sum(edges_inside.values()) // 2
    vol = sum(deg[n] for n in community)
    sem_sum_ns = sum(sim_map[n] for n in community if not is_seed[n])
    n_nonseed = sum(1 for n in community if not is_seed[n])
    cut = vol - 2 * n_int_edges
    current_energy = _sege_energy(sem_of(sem_sum_ns, n_nonseed), n_nodes, n_int_edges,
                                  vol, cut, total_graph_vol, lambda_bal, size_weight)
    improved, round_idx = True, 0
    while improved and len(community) > min_community_size and round_idx < 10:
        improved = False
        round_idx += 1
        best_remove, best_prune_e = None, current_energy
        for v in list(community):
            if v in valid_seeds and len(valid_seeds & community) <= 1:
                continue
            ei = edges_inside[v]
            dv = deg[v]
            d_sem = 0.0 if is_seed[v] else sim_map[v]
            d_ns = 0 if is_seed[v] else 1
            new_e = _sege_energy(sem_of(sem_sum_ns - d_sem, n_nonseed - d_ns),
                                 n_nodes - 1, n_int_edges - ei, vol - dv,
                                 (cut - dv) + 2 * ei, total_graph_vol, lambda_bal, size_weight)
            if new_e > best_prune_e + EPSILON:
                best_prune_e, best_remove = new_e, v
        if best_remove is None:
            break
        ei = edges_inside[best_remove]
        dv = deg[best_remove]
        community.remove(best_remove)
        edges_inside.pop(best_remove, None)
        n_nodes -= 1
        n_int_edges -= ei
        vol -= dv
        cut = (cut - dv) + 2 * ei
        if not is_seed[best_remove]:
            sem_sum_ns -= sim_map[best_remove]
            n_nonseed -= 1
        current_energy = best_prune_e
        for u in adj[best_remove]:
            if u in community:
                edges_inside[u] -= 1
        improved = True
    best_community = community

    if not (valid_seeds & best_community):
        return valid_seeds
    sub_G = subgraph.subgraph(list(best_community))
    if not nx.is_connected(sub_G):
        comps = list(nx.connected_components(sub_G))
        best_community = max(comps, key=lambda c: len(c & valid_seeds))
    if len(best_community) < min_community_size:
        fallback = set(valid_seeds)
        for s in valid_seeds:
            fallback.update(subgraph.neighbors(s))
        return fallback
    return best_community

def sege_self_distillation(subgraph: nx.Graph, subgraph_embedding: torch.Tensor,
                           query_seeds, iters: int = SDR_ITERS,
                           lambda_bal: float = 0.5, alpha: float = 0.7,
                           min_community_size: int = 3, max_community_size: int = 200,
                           anneal: float = SDR_ANNEAL, soft_beta: float = SDR_SOFT_BETA,
                           soft_tau: float = SDR_SOFT_TAU) -> Set[int]:
    seeds = {query_seeds} if isinstance(query_seeds, int) else set(query_seeds)
    if subgraph.number_of_nodes() == 0:
        return seeds
    valid_seeds = {s for s in seeds if s in subgraph.nodes()}
    if not valid_seeds:
        return seeds

    subgraph_nodes = sorted(list(subgraph.nodes()))
    node_to_idx = {n: i for i, n in enumerate(subgraph_nodes)}
    Z = subgraph_embedding
    if Z.shape[0] != len(subgraph_nodes):
        neighbors = set()
        for s in valid_seeds:
            neighbors.update(subgraph.neighbors(s))
        return neighbors | valid_seeds

    dev = Z.device
    seed_idx = torch.tensor([node_to_idx[s] for s in valid_seeds], device=dev)
    proto = Z.index_select(0, seed_idx).mean(dim=0)
    sharp = 1.0

    obs_edges0 = subgraph.number_of_edges()
    rho_deg = (2.0 * obs_edges0) / max(1, len(subgraph_nodes))
    structural_mode = False
    size_weight = 1.0 if max_community_size > 5000 else 0.3
    degree_weight = 0.8 if max_community_size > 5000 else 0.5

    CAND_BUDGET = max(4000, int(max_community_size))
    if len(subgraph_nodes) > CAND_BUDGET:
        if structural_mode:
            keep = set(sorted(subgraph_nodes, key=lambda n: -subgraph.degree(n))[:CAND_BUDGET])
        else:
            q0 = F.normalize(proto.reshape(1, -1), dim=1)
            z0 = F.normalize(Z, dim=1)
            sims0 = (F.cosine_similarity(q0, z0) + 1.0) / 2.0
            keep = {subgraph_nodes[i] for i in torch.topk(sims0, CAND_BUDGET).indices.cpu().tolist()}
        keep |= valid_seeds
        for s in valid_seeds:
            keep.update(subgraph.neighbors(s))
        keep_sorted = sorted(keep)
        Z = Z[torch.tensor([node_to_idx[n] for n in keep_sorted], device=dev)]
        subgraph = subgraph.subgraph(keep_sorted).copy()
        subgraph_nodes = sorted(list(subgraph.nodes()))
        node_to_idx = {n: i for i, n in enumerate(subgraph_nodes)}
        seed_idx = torch.tensor([node_to_idx[s] for s in valid_seeds if s in node_to_idx], device=dev)
        proto = Z.index_select(0, seed_idx).mean(dim=0)

    if structural_mode:
        lambda_bal = 0.85
    else:
        lambda_bal = float(0.2 + 0.6 * min(1.0, rho_deg / 15.0))
        try:
            z_norm = F.normalize(Z, dim=1)
            sim_mat = z_norm @ z_norm.t()
            sim_mat.fill_diagonal_(-1.0)
            k_sem = min(15, max(1, len(subgraph_nodes) - 1))
            topk = torch.topk(sim_mat, k_sem, dim=1).indices.cpu().numpy()
            subgraph.add_edges_from([(subgraph_nodes[i], subgraph_nodes[int(j)])
                                     for i, nbrs in enumerate(topk) for j in nbrs
                                     if subgraph_nodes[int(j)] != subgraph_nodes[i]])
        except Exception:
            pass

    C = sege_energy_search(subgraph, Z, valid_seeds, min_community_size, max_community_size,
                           lambda_bal, alpha, query_embedding=proto, sim_sharp=sharp, size_weight=size_weight, degree_weight=degree_weight)
    for _ in range(max(0, iters)):
        sharp = sharp / anneal
        cidx = [node_to_idx[n] for n in C if n in node_to_idx]
        if not cidx:
            break
        proto = Z[torch.tensor(cidx, device=dev)].mean(dim=0)
        C = sege_energy_search(subgraph, Z, valid_seeds, min_community_size,
                               max_community_size, lambda_bal, alpha,
                               query_embedding=proto, sim_sharp=sharp, size_weight=size_weight, degree_weight=degree_weight)

    if (not structural_mode) and rho_deg < 2.0:
        q = F.normalize(proto.reshape(1, -1), dim=1)
        z_norm = F.normalize(Z, dim=1)
        cos_all = F.cosine_similarity(q, z_norm).cpu().numpy()
        C = C | {subgraph_nodes[i] for i in np.where(cos_all > 0.0)[0].tolist()}
    return C | valid_seeds

def build_local_incident_graph(nodes: Set[int], snapshots: list, num_total_nodes: int) -> nx.Graph:
    local_nodes = {int(n) for n in nodes if n is not None}
    graph = nx.Graph()
    graph.add_nodes_from(local_nodes)
    if not local_nodes:
        return graph

    node_arr = np.fromiter(local_nodes, dtype=np.int64)
    for _, pyg_graph in snapshots:
        edge_index = getattr(pyg_graph, 'edge_index', None)
        if edge_index is None:
            continue
        ei = edge_index[0] if isinstance(edge_index, list) else edge_index
        ei_np = ei.cpu().numpy() if torch.is_tensor(ei) else np.asarray(ei)
        if ei_np.ndim != 2 or ei_np.shape[1] == 0:
            continue
        src = ei_np[0].astype(np.int64, copy=False)
        dst = ei_np[1].astype(np.int64, copy=False)
        mask = np.isin(src, node_arr) | np.isin(dst, node_arr)
        if not mask.any():
            continue
        graph.add_edges_from((int(u), int(v)) for u, v in zip(src[mask], dst[mask]))
    return graph

def _use_local_graph_scope(dataset_name: str, num_total_nodes: int) -> bool:
    local_profiles = {"stack", "dewiki", "bitcoin"}
    graph_scope = os.environ.get("GRAPH_SCOPE", "").strip().lower()
    if graph_scope in {"local", "bounded"}:
        return True
    if graph_scope in {"global", "full"}:
        return False
    threshold = int(os.environ.get("LOCAL_GRAPH_THRESHOLD", 5_000_000))
    return dataset_name in local_profiles or num_total_nodes >= threshold

def eds_semantic_ppr_sampling(query_node: int, all_snapshots: list,
                              m_current_idx: int, embedding_sequence_mgr,
                              partition_map, num_total_nodes: int,
                              aligned_gnn_inputs: list,
                              hops: int = 3, top_m: int = EDS_TOP_M,
                              teleport: float = EDS_TELEPORT, ppr_iters: int = EDS_ITERS,
                              temporal_lookback: int = EDS_LOOKBACK,
                              temporal_decay: float = EDS_DECAY):
    window_idx = list(range(max(0, m_current_idx - temporal_lookback + 1), m_current_idx + 1))
    ts_for_idx = {}
    for h in window_idx:
        try:
            tsv = aligned_gnn_inputs[h]['timestamp']
            ts_for_idx[h] = tsv.item() if torch.is_tensor(tsv) else tsv
        except Exception:
            continue
    needed = set(ts_for_idx.values())

    ts_to_graph = {}
    for item in all_snapshots:
        if isinstance(item, tuple) and len(item) == 2:
            tsv, g = item
        elif hasattr(item, 'timestamp'):
            tsv = item.timestamp
            tsv = tsv.item() if torch.is_tensor(tsv) else tsv
            g = item
        else:
            continue
        if tsv in needed:
            ts_to_graph[tsv] = g

    m_ts = ts_for_idx.get(m_current_idx)
    g_recent = ts_to_graph.get(m_ts)
    if g_recent is None:
        return [query_node], torch.tensor([[query_node], [query_node]])

    e_recent = g_recent.edge_index[0] if isinstance(g_recent.edge_index, list) else g_recent.edge_index
    e_recent = to_undirected(e_recent, num_nodes=g_recent.num_nodes)
    support, _, _, _ = k_hop_subgraph(query_node, hops, e_recent,
                                      num_nodes=g_recent.num_nodes, relabel_nodes=False)
    sup_list = sorted(set(support.cpu().tolist()) | {query_node})
    N = len(sup_list)
    pos = {n: i for i, n in enumerate(sup_list)}

    raw_m = embedding_sequence_mgr[m_current_idx]
    emb_super = raw_m.get('embedding', None)
    part_arr = np.asarray(partition_map)
    z_norm = None
    if emb_super is not None and hasattr(emb_super, 'shape') and part_arr.size > 0:
        if sp.issparse(emb_super):
            emb_super = emb_super.toarray()
        emb_super = np.asarray(emb_super, dtype=np.float32)
        part_sub = part_arr[sup_list]
        valid = part_sub < emb_super.shape[0]
        z_fine = np.zeros((N, emb_super.shape[1]), dtype=np.float32)
        z_fine[valid] = emb_super[part_sub[valid]]
        z_norm = F.normalize(torch.from_numpy(z_fine).float(), dim=1)

    sup_arr = np.asarray(sup_list, dtype=np.int64)

    rows, cols, vals = [], [], []
    for h in sorted(window_idx):
        g = ts_to_graph.get(ts_for_idx.get(h))
        if g is None:
            continue
        ei_g = g.edge_index[0] if isinstance(g.edge_index, list) else g.edge_index
        ei_np = ei_g.cpu().numpy() if torch.is_tensor(ei_g) else np.asarray(ei_g)
        s, d = ei_np[0].astype(np.int64), ei_np[1].astype(np.int64)
        in_sup = np.isin(s, sup_arr) & np.isin(d, sup_arr)
        if not in_sup.any():
            continue
        s_keep, d_keep = s[in_sup], d[in_sup]
        up = np.searchsorted(sup_arr, s_keep)
        vp = np.searchsorted(sup_arr, d_keep)
        in_range = (up < N) & (vp < N)
        valid = np.zeros_like(in_range, dtype=bool)
        if in_range.any():
            valid[in_range] = (sup_arr[up[in_range]] == s_keep[in_range]) & (
                sup_arr[vp[in_range]] == d_keep[in_range]
            )
        if not valid.any():
            continue
        up, vp = up[valid], vp[valid]
        decay_w = float(temporal_decay ** (m_current_idx - h))
        rows.append(up); cols.append(vp); vals.append(np.full(up.shape, decay_w))
        rows.append(vp); cols.append(up); vals.append(np.full(up.shape, decay_w))

    if rows:
        R = np.concatenate(rows); Cc = np.concatenate(cols); V = np.concatenate(vals)
    else:
        R = Cc = V = np.zeros(0, dtype=np.int64)
    Wsp = sp.csc_matrix((V.astype(np.float64), (R, Cc)), shape=(N, N)) if R.size > 0 \
        else sp.csc_matrix((N, N))

    if z_norm is not None and Wsp.nnz > 0:
        coo = Wsp.tocoo()
        sem = ((z_norm[torch.from_numpy(coo.row.astype(np.int64))] *
                z_norm[torch.from_numpy(coo.col.astype(np.int64))]).sum(dim=1) + 1.0) / 2.0
        sem = torch.clamp(sem, min=1e-3).numpy().astype(np.float64)
        Wsp = sp.csc_matrix((coo.data * sem, (coo.row, coo.col)), shape=(N, N))

    deg = np.asarray(Wsp.sum(axis=1)).ravel()
    dinv = np.zeros(N)
    nz = deg > 0
    dinv[nz] = 1.0 / deg[nz]
    Tt = (sp.diags(dinv) @ Wsp).T.tocsr()
    e_q = np.zeros(N)
    e_q[pos.get(query_node, 0)] = 1.0
    p = e_q.copy()
    for _ in range(ppr_iters):
        p = teleport * e_q + (1.0 - teleport) * (Tt @ p)

    if p.max() <= 0:
        roi_set = set(sup_list)
    else:
        keep = min(top_m, N)
        top_idx = np.argpartition(p, -keep)[-keep:] if keep < N else np.arange(N)
        roi_set = {sup_list[i] for i in top_idx}
    roi_set.add(query_node)
    qn = query_node
    nb_mask = (e_recent[0] == qn) if torch.is_tensor(e_recent) else np.asarray(e_recent[0]) == qn
    for nb in (e_recent[1][nb_mask].tolist() if torch.is_tensor(e_recent) else np.asarray(e_recent[1])[nb_mask].tolist()):
        if int(nb) in pos:
            roi_set.add(int(nb))

    roi_nodes = sorted(roi_set)

    roi_arr = np.asarray(roi_nodes, dtype=np.int64)
    er = e_recent.cpu().numpy() if torch.is_tensor(e_recent) else np.asarray(e_recent)
    m_keep = np.isin(er[0].astype(np.int64), roi_arr) & np.isin(er[1].astype(np.int64), roi_arr)
    if m_keep.any():
        obs_ei = torch.tensor(np.stack([er[0][m_keep], er[1][m_keep]]), dtype=torch.long)
    else:
        obs_ei = torch.tensor([[query_node], [query_node]], dtype=torch.long)
    return roi_nodes, obs_ei

def reconstruct_structure_from_embedding(
    nodes_in_community: Set[int],
    full_local_embeddings: torch.Tensor,
    full_subgraph_node_list: List[int],
    similarity_metric: str = 'cosine',
    reference_avg_degree: float = None,
    dynamic_std_scale: float = 1.0,
    hard_threshold: float = None
) -> nx.Graph:
    comm_nodes_list = sorted(list(nodes_in_community))
    N = len(comm_nodes_list)
    if N == 0: return nx.Graph()

    node_to_idx_full = {node_id: i for i, node_id in enumerate(full_subgraph_node_list)}
    valid_indices = []
    valid_nodes = []
    for node_id in comm_nodes_list:
        if node_id in node_to_idx_full:
            valid_indices.append(node_to_idx_full[node_id])
            valid_nodes.append(node_id)

    if not valid_indices:
        g = nx.Graph()
        g.add_nodes_from(comm_nodes_list)
        return g

    device = full_local_embeddings.device
    indices_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
    z_sub = full_local_embeddings.index_select(0, indices_tensor)

    if similarity_metric == 'cosine':
        z_sub_norm = F.normalize(z_sub, p=2, dim=1)
        sim_matrix = torch.mm(z_sub_norm, z_sub_norm.t())
    else:
        sim_matrix = torch.mm(z_sub, z_sub.t())

    mask_off_diag = ~torch.eye(z_sub.shape[0], dtype=torch.bool, device=device)
    valid_sims = sim_matrix[mask_off_diag]

    if valid_sims.numel() == 0:
        predicted_graph = nx.Graph()
        predicted_graph.add_nodes_from(valid_nodes)
        return predicted_graph

    mean_sim = valid_sims.mean()
    std_sim = valid_sims.std()
    stat_threshold = mean_sim + dynamic_std_scale * std_sim

    k_threshold = -1.0
    if reference_avg_degree is not None:
        target_avg_deg = max(reference_avg_degree, 1.0)
        target_num_edges = int((N * target_avg_deg * 1.2) // 2)
        target_num_edges = max(target_num_edges, N - 1)
        max_possible_edges = valid_sims.numel() // 2
        target_num_edges = min(target_num_edges, max_possible_edges)

        if target_num_edges > 0:
            k_top = min(valid_sims.numel(), target_num_edges * 2)
            top_val, _ = torch.topk(valid_sims.flatten(), k_top)
            k_threshold = top_val[-1]

    final_threshold = max(stat_threshold, k_threshold)
    final_threshold = max(final_threshold, 0.1)

    if hard_threshold is not None:
        final_threshold = hard_threshold

    adj_mask = (sim_matrix > final_threshold) & mask_off_diag
    del sim_matrix, valid_sims

    edge_indices = adj_mask.nonzero(as_tuple=False)
    predicted_graph = nx.Graph()
    predicted_graph.add_nodes_from(valid_nodes)

    edge_list = []
    edge_indices_cpu = edge_indices.cpu().numpy()

    for r, c in edge_indices_cpu:
        u = valid_nodes[r]
        v = valid_nodes[c]
        if u < v:
            edge_list.append((u, v))

    predicted_graph.add_edges_from(edge_list)
    return predicted_graph

def refine_online_graph_with_embeddings(
    graph: nx.Graph,
    node_embeddings: torch.Tensor,
    node_order: List[int],
    query_node: int,
    max_refine_nodes: int = 5000,
    top_k: int = 6,
    min_similarity: float = 0.25,
    chunk_size: int = 512,
) -> nx.Graph:
    if graph.number_of_nodes() <= 1 or node_embeddings is None:
        return graph
    if len(node_order) != int(node_embeddings.shape[0]) or top_k <= 0:
        return graph

    n_total = len(node_order)
    if n_total <= max_refine_nodes:
        selected_nodes = list(node_order)
    else:
        selected = {int(query_node)} if query_node in graph else set()
        degree_rank = sorted(node_order, key=lambda n: graph.degree(n), reverse=True)
        for n in degree_rank:
            selected.add(int(n))
            if len(selected) >= max_refine_nodes:
                break
        selected_nodes = sorted(selected)

    if len(selected_nodes) <= 1:
        return graph

    node_to_idx = {n: i for i, n in enumerate(node_order)}
    selected_indices = [node_to_idx[n] for n in selected_nodes if n in node_to_idx]
    if len(selected_indices) <= 1:
        return graph

    with torch.no_grad():
        z = node_embeddings.index_select(
            0, torch.tensor(selected_indices, dtype=torch.long, device=node_embeddings.device)
        ).float()
        z = F.normalize(z, dim=1)
        k_eff = min(top_k, z.shape[0] - 1)
        if k_eff <= 0:
            return graph
        added_edges = []
        for start in range(0, z.shape[0], chunk_size):
            end = min(start + chunk_size, z.shape[0])
            sims = z[start:end] @ z.t()
            row_ids = torch.arange(start, end, device=sims.device)
            sims[torch.arange(end - start, device=sims.device), row_ids] = -1.0
            vals, nbrs = torch.topk(sims, k=k_eff, dim=1)
            vals_cpu = vals.cpu().numpy()
            nbrs_cpu = nbrs.cpu().numpy()
            for local_i, src_pos in enumerate(range(start, end)):
                u = int(selected_nodes[src_pos])
                for val, dst_pos in zip(vals_cpu[local_i], nbrs_cpu[local_i]):
                    if float(val) < min_similarity:
                        continue
                    v = int(selected_nodes[int(dst_pos)])
                    if u != v:
                        added_edges.append((u, v))
        if added_edges:
            graph.add_edges_from(added_edges)
    return graph

EDGE_DEL_THRESHOLD = 0.10

def delete_low_similarity_edges(graph: nx.Graph,
                                node_embeddings: torch.Tensor,
                                node_order: List[int],
                                delta_del: float = EDGE_DEL_THRESHOLD) -> nx.Graph:
    if graph.number_of_edges() == 0 or node_embeddings is None:
        return graph
    node_to_idx = {n: i for i, n in enumerate(node_order)}
    elist = [(int(u), int(v)) for u, v in graph.edges()
             if u in node_to_idx and v in node_to_idx]
    if not elist:
        return graph
    with torch.no_grad():
        z = F.normalize(node_embeddings.float(), dim=1)
        idx = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in elist],
                           dtype=torch.long, device=node_embeddings.device)
        sims = (z[idx[:, 0]] * z[idx[:, 1]]).sum(dim=1)
        mask = sims < delta_del
        remove = [elist[i] for i in range(len(elist)) if bool(mask[i])]
        if remove:
            graph.remove_edges_from(remove)
    return graph

def compute_mmd(x: np.ndarray, y: np.ndarray, sigma: float = None) -> float:
    if len(x) == 0 or len(y) == 0: return 0.0
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)

    if sigma is None:
        z = np.vstack((x, y))
        if z.shape[0] > 2000:
            idx = np.random.choice(z.shape[0], 2000, replace=False)
            z_sample = z[idx]
        else:
            z_sample = z
        dists = euclidean_distances(z_sample, z_sample)
        upper_tri = dists[np.triu_indices(dists.shape[0], k=1)]
        sigma = float(np.median(upper_tri)) if len(upper_tri) > 0 else 1.0
        if sigma < 1e-6: sigma = 1.0

    gamma = 1.0 / (2 * sigma**2)
    xx = np.exp(-gamma * euclidean_distances(x, x, squared=True))
    yy = np.exp(-gamma * euclidean_distances(y, y, squared=True))
    xy = np.exp(-gamma * euclidean_distances(x, y, squared=True))
    mmd_sq = xx.mean() + yy.mean() - 2 * xy.mean()
    return np.sqrt(max(mmd_sq, 0.0))

def evaluate_community_quality(found_community: Set[int], graph: nx.Graph, time_window: int = 1):
    S = found_community
    metrics = {"Community Nodes": len(S), "Community Edges": 0, "Density": 0.0, "Conductance": 0.0, "TC": 0.0, "TD": 0.0}
    if not S or len(S) < 2: return metrics

    internal_edges = graph.subgraph(S).size()
    metrics["Community Edges"] = internal_edges
    denominator_td = len(S) * (len(S) - 1) * time_window
    metrics["TD"] = (2 * internal_edges) / denominator_td if denominator_td > 0 else 0.0
    metrics["Density"] = metrics["TD"]

    try:
        vol_S = sum(graph.degree[n] for n in S)
        cut_S = vol_S - 2 * internal_edges
        if cut_S == 0:
            metrics["TC"] = 0.0
            metrics["Conductance"] = 0.0
        else:
            total_vol = 2 * graph.number_of_edges()
            denominator_tc = min(vol_S, total_vol - vol_S)
            metrics["TC"] = cut_S / denominator_tc if denominator_tc > 0 else float('inf')
            metrics["Conductance"] = metrics["TC"]
    except:
        metrics["TC"] = float('inf')
        metrics["Conductance"] = float('inf')
    return metrics

def compute_set_metrics(pred_nodes, gt_nodes, universe_nodes):
    if not pred_nodes or not gt_nodes:
        return {"F1": 0.0, "Precision": 0.0, "Recall": 0.0, "Jaccard": 0.0, "NMI": 0.0}

    intersection = len(pred_nodes & gt_nodes)
    precision = intersection / len(pred_nodes)
    recall = intersection / len(gt_nodes)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = intersection / len(pred_nodes | gt_nodes)

    all_relevant = list(universe_nodes | gt_nodes)
    y_true = [1 if n in gt_nodes else 0 for n in all_relevant]
    y_pred = [1 if n in pred_nodes else 0 for n in all_relevant]
    nmi = normalized_mutual_info_score(y_true, y_pred)

    return {"F1": f1, "Precision": precision, "Recall": recall, "Jaccard": jaccard, "NMI": nmi}

def run_combined_pipeline(
    diffusion_model, refinement_gnn,
    all_snapshots: list,
    test_snapshots: list,
    all_data_for_diffusion: dict,
    t_future_idx: int,
    m_current_idx: int,
    query_node: int,
    hops: int,
    device: str,
    dataset_name: str,
    history_window_size: int = 5,
    guidance_scale: float = 2.0,
    diffusion_steps: int = 20,
    num_total_nodes: int = 0,
    pool_budget_sn: int = 300,
    candidate_nodes=None,
    future_edge_index=None,
    future_num_nodes: int = 0,
    max_out_pool: int = 6000,
):
    timings = {}
    total_start_time = time.time()

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_region = time.time()

    embedding_sequence_mgr = all_data_for_diffusion['embedding_sequence']

    if candidate_nodes is not None:
        roi_nodes = sorted(int(x) for x in candidate_nodes)
        observed_edge_index = future_edge_index
    elif future_edge_index is not None:
        fut_ei = future_edge_index
        if not torch.is_tensor(fut_ei):
            fut_ei = torch.as_tensor(fut_ei, dtype=torch.long)
        try:
            fh_nodes, fh_edge_index, _, _ = k_hop_subgraph(
                torch.tensor([int(query_node)], dtype=torch.long),
                num_hops=hops, edge_index=fut_ei,
                relabel_nodes=False, num_nodes=int(future_num_nodes)
            )
            roi_nodes = [int(x) for x in fh_nodes.cpu().tolist()]
            observed_edge_index = fh_edge_index if fh_edge_index.numel() > 0 else None
        except Exception:
            roi_nodes = [int(query_node)]
            observed_edge_index = None
    else:
        partition_map_m = all_data_for_diffusion['partition_history'][m_current_idx]
        roi_nodes, observed_edge_index = eds_semantic_ppr_sampling(
            query_node=query_node,
            all_snapshots=all_snapshots,
            m_current_idx=m_current_idx,
            embedding_sequence_mgr=embedding_sequence_mgr,
            partition_map=partition_map_m,
            num_total_nodes=num_total_nodes,
            aligned_gnn_inputs=all_data_for_diffusion['aligned_gnn_inputs'],
            hops=hops,
        )

    predicted_subgraph_nx = nx.Graph()
    predicted_subgraph_nx.add_nodes_from(roi_nodes)
    if torch.is_tensor(observed_edge_index) and observed_edge_index.numel() > 0:
        oe = observed_edge_index.cpu().numpy()
        predicted_subgraph_nx.add_edges_from(list(zip(oe[0].tolist(), oe[1].tolist())))

    timings['Time_Region'] = time.time() - start_region

    nodes_in_subgraph = sorted(list(predicted_subgraph_nx.nodes()))
    if not nodes_in_subgraph: nodes_in_subgraph = [query_node]

    embedding_sequence_mgr = all_data_for_diffusion['embedding_sequence']
    current_adj_sparse = all_data_for_diffusion['aligned_gnn_inputs'][m_current_idx]['adj']
    partition_map = all_data_for_diffusion['partition_history'][m_current_idx]

    valid_nodes = [n for n in nodes_in_subgraph if n < len(partition_map)]
    if not valid_nodes: valid_nodes = [query_node]

    relevant_blocks_cpu = np.unique([partition_map[n] for n in valid_nodes])
    relevant_blocks_tensor_cpu = torch.from_numpy(relevant_blocks_cpu).long()

    num_coarse_nodes = current_adj_sparse.shape[0]
    coarse_edge_index, _ = from_scipy_sparse_matrix(current_adj_sparse)

    pool_budget = max(50, min(int(pool_budget_sn), int(0.4 * num_coarse_nodes)))
    q_super = int(relevant_blocks_cpu[0]) if len(relevant_blocks_cpu) > 0 else 0
    sn_set = set()
    for _h in (1, 2, 3, 4, 5):
        s, _, _, _ = k_hop_subgraph(relevant_blocks_tensor_cpu, num_hops=_h,
                                    edge_index=coarse_edge_index, relabel_nodes=False,
                                    num_nodes=num_coarse_nodes)
        sn_set = set(int(x) for x in s.cpu().tolist())
        if len(sn_set) >= pool_budget:
            break
    if len(sn_set) < pool_budget:
        try:
            raw_obs = embedding_sequence_mgr[m_current_idx]
            obs_emb = raw_obs.get('embedding', None)
            if obs_emb is not None and hasattr(obs_emb, 'shape') and obs_emb.shape[0] > 1:
                if sp.issparse(obs_emb):
                    obs_emb = obs_emb.toarray()
                obs_emb = np.asarray(obs_emb, dtype=np.float32)
                if 0 <= q_super < obs_emb.shape[0]:
                    obs_n = F.normalize(torch.from_numpy(obs_emb), dim=1)
                    sims_s = (obs_n @ obs_n[q_super]).cpu().numpy()
                    for i in np.argsort(-sims_s):
                        sn_set.add(int(i))
                        if len(sn_set) >= pool_budget:
                            break
        except Exception:
            pass
    sn_list = sorted(sn_set)[:pool_budget]
    if 0 <= q_super < num_coarse_nodes and q_super not in sn_list:
        sn_list = sorted(set(sn_list) | {q_super})
    new_sn_tensor = torch.tensor(sn_list, dtype=torch.long)
    cei = coarse_edge_index
    emask = torch.isin(cei[0], new_sn_tensor) & torch.isin(cei[1], new_sn_tensor)
    e0 = torch.searchsorted(new_sn_tensor, cei[0][emask])
    e1 = torch.searchsorted(new_sn_tensor, cei[1][emask])
    sub_edge_index = torch.stack([e0, e1], dim=0)
    pos = {s: i for i, s in enumerate(sn_list)}
    mapping = torch.tensor([pos[int(b)] for b in relevant_blocks_cpu.tolist() if int(b) in pos],
                           dtype=torch.long)
    if mapping.numel() == 0:
        mapping = torch.zeros(1, dtype=torch.long)
    subset = new_sn_tensor

    computation_blocks = subset.to(device)
    sub_edge_index = sub_edge_index.to(device)
    target_indices_in_subset = mapping.to(device)

    start_idx = max(0, m_current_idx - history_window_size + 1)
    history_indices = list(range(start_idx, m_current_idx + 1))

    local_hist_list = []
    for h_idx in history_indices:
        raw_item = embedding_sequence_mgr[h_idx]
        local_emb = extract_local_embedding(raw_item, subset, device, num_total_nodes)
        local_hist_list.append(local_emb)

    hist_embs_sub = torch.stack(local_hist_list).permute(1, 0, 2).to(device)

    curr_ts_val = embedding_sequence_mgr[m_current_idx]['timestamp']
    hist_ts_vals = torch.tensor([embedding_sequence_mgr[i]['timestamp'] for i in history_indices])
    hist_rel_ts_sub = (hist_ts_vals - curr_ts_val).unsqueeze(0).expand(len(computation_blocks), -1).to(device)

    delta_t_val = t_future_idx - m_current_idx
    delta_t_sub = torch.tensor([delta_t_val], device=device).long().expand(len(computation_blocks))
    x_rand_sub = torch.randn(len(computation_blocks), hist_embs_sub.shape[-1], device=device)

    diffusion = Diffusion(timesteps=1000, device=device)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_diff = time.time()

    with torch.no_grad():
        with auto_cast_ctx():
            _, x_c_sub = diffusion_model(
                x_rand_sub, torch.zeros(len(computation_blocks), device=device).long(),
                hist_embs_sub, hist_rel_ts_sub, delta_t_sub, edge_index=sub_edge_index
            )
            refined_blocks_all = diffusion.ddim_sample_loop(
                diffusion_model, shape=(len(computation_blocks), hist_embs_sub.shape[-1]),
                history_embeddings=hist_embs_sub, history_relative_timestamps=hist_rel_ts_sub,
                delta_t=delta_t_sub, x_c=x_c_sub, edge_index=sub_edge_index,
                num_steps=diffusion_steps, guidance_scale=guidance_scale,
                show_progress=False
            )

    subset_cpu = computation_blocks.cpu().tolist()
    refined_all_cpu = refined_blocks_all.cpu()
    coarse_emb_map = {int(subset_cpu[i]): refined_all_cpu[i] for i in range(len(subset_cpu))}

    del hist_embs_sub, x_rand_sub, x_c_sub, refined_blocks_all

    if torch.cuda.is_available(): torch.cuda.synchronize()
    timings['Time_Diffusion'] = time.time() - start_diff

    start_ref = time.time()
    part_arr = np.asarray(partition_map)
    subset_set = list(set(subset_cpu))
    if len(subset_set) > 0:
        coarse_fine = np.where(np.isin(part_arr, np.asarray(subset_set,
                                                            dtype=part_arr.dtype)))[0]
    else:
        coarse_fine = np.array([], dtype=np.int64)

    broad_nodes = set(nodes_in_subgraph) | {int(x) for x in coarse_fine.tolist()}
    broad_nodes.add(query_node)
    MAX_FINE = max_out_pool
    if len(broad_nodes) > MAX_FINE:
        keep = set(nodes_in_subgraph) | {query_node}
        for x in coarse_fine.tolist():
            if len(keep) >= MAX_FINE:
                break
            keep.add(int(x))
        broad_nodes = keep
    nodes_in_subgraph = sorted(broad_nodes)

    predicted_subgraph_nx = nx.Graph()
    predicted_subgraph_nx.add_nodes_from(nodes_in_subgraph)
    _bn = set(nodes_in_subgraph)
    if torch.is_tensor(observed_edge_index) and observed_edge_index.numel() > 0:
        _oe = observed_edge_index.cpu().numpy()
        predicted_subgraph_nx.add_edges_from(
            [(int(u), int(v)) for u, v in zip(_oe[0], _oe[1])
             if int(u) in _bn and int(v) in _bn])
    dim = refined_all_cpu.shape[-1]
    uplifted_local_list = []
    zero_vec = torch.zeros(dim)
    for n in nodes_in_subgraph:
        if n < part_arr.shape[0]:
            blk = int(part_arr[n])
            uplifted_local_list.append(coarse_emb_map.get(blk, zero_vec))
        else:
            uplifted_local_list.append(zero_vec)
    uplifted_local = torch.stack(uplifted_local_list).to(device)

    node_map_fine = {n: i for i, n in enumerate(nodes_in_subgraph)}
    local_edges = [
        (node_map_fine[int(u)], node_map_fine[int(v)])
        for u, v in predicted_subgraph_nx.edges()
        if int(u) in node_map_fine and int(v) in node_map_fine
    ]
    if local_edges:
        local_edge_index = torch.tensor(local_edges, dtype=torch.long, device=device).t().contiguous()
        local_edge_index = to_undirected(local_edge_index, num_nodes=len(nodes_in_subgraph))
    else:
        loop = torch.arange(len(nodes_in_subgraph), dtype=torch.long, device=device)
        local_edge_index = torch.stack([loop, loop], dim=0)

    with torch.no_grad():
        with auto_cast_ctx():
            final_refined_local = refinement_gnn(uplifted_local, local_edge_index)

    predicted_subgraph_nx = delete_low_similarity_edges(
        predicted_subgraph_nx,
        final_refined_local,
        nodes_in_subgraph,
    )

    predicted_subgraph_nx = refine_online_graph_with_embeddings(
        predicted_subgraph_nx,
        final_refined_local,
        nodes_in_subgraph,
        query_node=query_node,
    )

    timings['Time_Refinement'] = time.time() - start_ref
    timings['Time_Total'] = time.time() - total_start_time

    return final_refined_local.cpu(), predicted_subgraph_nx, query_node, timings

def load_community_ground_truth(query_file):
    queries = []
    node_to_comm = {}
    comm_to_nodes = {}

    if not os.path.exists(query_file):
        return queries, node_to_comm, comm_to_nodes

    with open(query_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    is_format_a = ':' in first_line

    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if is_format_a:
                    header, members_str = line.split(':', 1)
                    node_time_str = header.split(',')
                    query_node = int(node_time_str[0])
                    timestamp = int(node_time_str[1])
                    community = set(int(x) for x in members_str.split() if x.strip())
                    queries.append({
                        "query_node": query_node,
                        "timestamp": timestamp,
                        "community": community
                    })
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        node_id = int(parts[0])
                        comm_id = int(parts[1])
                        node_to_comm[node_id] = comm_id
                        comm_to_nodes.setdefault(comm_id, set()).add(node_id)
            except (ValueError, IndexError):
                continue

    return queries, node_to_comm, comm_to_nodes

def build_gt_lookup(gt_queries, ts_set):
    lookup = {}
    for q in gt_queries:
        ts = q["timestamp"]
        if ts in ts_set:
            lookup.setdefault(ts, []).append(q)
    return lookup
def main_multi_round_evaluation(
    diffusion_model_path, refinement_model_path,
    device, dataset_name, p1_pkl_path, p2_coarse_pkl_path,
    diff_steps, num_queries, query_size, alpha
):
    print(f"\n[Info] Starting Optimized Evaluation for {dataset_name}...")
    print_memory_usage("Start")

    try:
        p1_data = joblib.load(p1_pkl_path)
    except:
        with open(p1_pkl_path, 'rb') as f: p1_data = pickle.load(f)

    num_total_nodes = p1_data['num_total_nodes']
    train_gnn = p1_data.get('train_aligned_inputs', [])
    val_gnn = p1_data.get('val_aligned_inputs', [])
    test_gnn = p1_data.get('test_aligned_inputs', [])

    train_part = p1_data.get('train_partition_history', [])
    val_part = p1_data.get('val_partition_history', [])
    test_part = p1_data.get('test_partition_history', [])

    full_gnn_seq = train_gnn + val_gnn + test_gnn

    if isinstance(train_part, list):
        full_part_seq = train_part + val_part + (list(test_part) if test_part is not None else [])
    else:
        parts = [train_part, val_part]
        if test_part is not None: parts.append(test_part)
        try:
            full_part_seq = np.concatenate(parts, axis=0)
        except:
            full_part_seq = list(train_part) + list(val_part) + list(test_part)

    del p1_data
    gc.collect()
    print_memory_usage("After P1 Load")

    disk_data_mgr = DiskBasedDataManager(p2_coarse_pkl_path, cache_dir_base=os.path.join(PROJECT_ROOT, "cache", dataset_name))

    _graph_dir = os.environ.get('GRAPH_DIR', os.path.join(PROJECT_ROOT, "datasets", dataset_name))
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        all_snapshots_pyg, _ = load_temporal_graph(_graph_dir)

    temp_snapshots = []
    for item in all_snapshots_pyg:
        if isinstance(item, tuple) and len(item) == 2:
            temp_snapshots.append(item)
        elif hasattr(item, 'timestamp'):
            val = item.timestamp
            if torch.is_tensor(val): val = val.item()
            temp_snapshots.append((val, item))

    needed_ts = {item['timestamp'] for item in full_gnn_seq}
    snapshots = sorted([x for x in temp_snapshots if x[0] in needed_ts], key=lambda x: x[0])
    ts_to_pyg_map = {ts: g for ts, g in snapshots}

    print_memory_usage("After Graph Load")

    detected_dim = 32
    if len(disk_data_mgr) > 0:
        sample_emb = disk_data_mgr[0]['embedding']
        if sample_emb is not None:
             detected_dim = sample_emb.shape[1] if len(sample_emb.shape) > 1 else 1

    diff_model = BlockDenoisingModelV14(embed_dim=detected_dim, time_embed_dim=32).to(device)
    diff_model.load_state_dict(torch.load(diffusion_model_path, map_location=device))
    diff_model.eval()

    ref_model = EmbeddingRefinementGNN_V15_1(embed_dim=detected_dim).to(device)
    ref_model.load_state_dict(torch.load(refinement_model_path, map_location=device))
    ref_model.eval()

    m_anchor_idx = len(train_gnn) + len(val_gnn) - 1
    query_file = os.path.join(PROJECT_ROOT, "datasets", f"{dataset_name}_gt.txt")

    gt_queries, node_to_comm, comm_to_nodes = load_community_ground_truth(query_file)
    has_format_a = len(gt_queries) > 0
    has_format_b = len(node_to_comm) > 0
    has_ground_truth = has_format_a or has_format_b

    gt_lookup = {}
    usable_gt_timestamps = set()
    if has_format_a:
        gt_lookup = build_gt_lookup(gt_queries, needed_ts)
        usable_gt_timestamps = set(gt_lookup.keys())
        print(f"[Info] GT Format A (query): {len(gt_queries)} community queries across {len(usable_gt_timestamps)} time steps.")
    elif has_format_b:
        print(f"[Info] GT Format B (partition): {len(node_to_comm)} nodes in {len(comm_to_nodes)} communities.")
    else:
        print(f"[Info] No ground truth file found for {dataset_name}. Only Density/Conductance will be reported.")

    diffusion_data = {
        'embedding_sequence': disk_data_mgr,
        'partition_history': full_part_seq,
        'aligned_gnn_inputs': full_gnn_seq,
        'num_total_nodes': num_total_nodes
    }

    _obs_ts = {full_gnn_seq[h]['timestamp'] for h in range(m_anchor_idx + 1)}
    obs_edge_count = int(sum(
        gg.edge_index.shape[1] for ts, gg in snapshots if ts in _obs_ts and hasattr(gg, 'edge_index')
    ))
    obs_avg_degree = (2.0 * obs_edge_count / max(1, num_total_nodes))
    num_super_est = max(1, int(num_total_nodes * 0.05))
    super_size = num_total_nodes / num_super_est

    if num_total_nodes <= 10000:
        max_out = int(max(50, min(0.20 * num_total_nodes, 0.4 * num_total_nodes)))
    elif num_total_nodes <= 250000:
        max_out = int(max(1000, min(0.075 * num_total_nodes, 50000)))
    elif num_total_nodes <= 1000000:
        max_out = int(max(8000, min(0.04 * num_total_nodes, 50000)))
    else:
        max_out = int(max(12000, min(0.02 * num_total_nodes, 80000)))
    if obs_avg_degree < 2.0:
        max_out = int(max(50, max_out * 0.8))

    pool_budget_sn = int(max(50, min(2.0 * max_out / super_size, 0.4 * num_super_est)))
    print(f"[Info] |V|={num_total_nodes}, |E_obs|={obs_edge_count}, "
          f"avg_deg_obs={obs_avg_degree:.2f}, max_out={max_out}, "
          f"pool budget={pool_budget_sn} sn")

    local_graph_scope = _use_local_graph_scope(dataset_name, num_total_nodes)
    fut_cum_ei_np = None
    fut_comm_sets = []
    fut_n2comm = {}
    true_cum_G = None

    if local_graph_scope:
        print("[Info] Local graph scope enabled for query-time structure and evaluation.")
    else:
        try:
            from networkx.algorithms.community import louvain_communities
            if len(snapshots) > 0:
                _obs_cum = torch.unique(torch.cat(
                    [gg.edge_index for ts, gg in snapshots if ts in _obs_ts], dim=1), dim=1)
                fut_cum_ei_np = _obs_cum.cpu().numpy()
                _obsG = nx.Graph(); _obsG.add_nodes_from(range(num_total_nodes))
                _obsG.add_edges_from(zip(fut_cum_ei_np[0].tolist(), fut_cum_ei_np[1].tolist()))
                _comms = louvain_communities(_obsG, seed=0)
                fut_comm_sets = [set(int(x) for x in c) for c in _comms]
                for _i, _c in enumerate(fut_comm_sets):
                    for _n in _c:
                        fut_n2comm[_n] = _i
                print(f"[Info] Observed accumulated graph (train+val, no leakage) Louvain: {len(fut_comm_sets)} communities; E={_obsG.number_of_edges()}")
                del _obsG

                _all_cum = torch.unique(torch.cat([gg.edge_index for _, gg in snapshots], dim=1), dim=1)
                _ac = _all_cum.cpu().numpy()
                true_cum_G = nx.Graph(); true_cum_G.add_nodes_from(range(num_total_nodes))
                true_cum_G.add_edges_from(zip(_ac[0].tolist(), _ac[1].tolist()))
                del _obs_cum, _all_cum
        except Exception as _e:
            print(f"[Warn] Global graph precompute failed ({_e}); falling back to local graph scope.")
            local_graph_scope = True
            fut_cum_ei_np = None
            fut_comm_sets = []
            fut_n2comm = {}
            true_cum_G = None

    test_start = len(train_gnn) + len(val_gnn)
    test_end = len(full_gnn_seq) - 1
    available_ts_indices = [i for i in range(test_start, test_end + 1) if full_gnn_seq[i]['timestamp'] in ts_to_pyg_map]

    if has_format_a:
        available_ts_indices = [i for i in available_ts_indices if full_gnn_seq[i]['timestamp'] in usable_gt_timestamps]

    results = []

    target_count = int(os.environ.get('TARGET_COUNT', num_queries))
    current_count = 0
    tried_queries = set()

    exhausted_gt = False

    print(f"\n[Info] Processing Queries (Target: {target_count})...")

    _deg_cache = {}
    _consecutive_empty = 0
    _max_consecutive_empty = len(available_ts_indices) * 15

    while current_count < target_count:
        if not available_ts_indices: break

        t_idx = random.choice(available_ts_indices)
        t_ts = full_gnn_seq[t_idx]['timestamp']
        curr_data = ts_to_pyg_map[t_ts]

        if curr_data.num_nodes == 0: continue

        if has_format_a:
            gt_entries = gt_lookup.get(t_ts, [])
            if not gt_entries:
                continue

            untried_entries = [e for e in gt_entries if (e["query_node"], t_idx) not in tried_queries]
            if not untried_entries:
                _consecutive_empty += 1
                if _consecutive_empty > _max_consecutive_empty:
                    print(f"[Info] Exhausted all available queries after {current_count} results.")
                    break
                continue

            gt_entry = random.choice(untried_entries)
            q_node = gt_entry["query_node"]
            gt_nodes = gt_entry["community"]

            tried_queries.add((q_node, t_idx))

        elif has_format_b:
            available_gt_nodes = [n for n in node_to_comm.keys() if n < curr_data.num_nodes]
            if not available_gt_nodes:
                continue

            candidate_nodes = available_gt_nodes
            untried = [n for n in candidate_nodes if (n, t_idx) not in tried_queries]
            if not untried:
                _consecutive_empty += 1
                if _consecutive_empty > _max_consecutive_empty:
                    print(f"[Info] Exhausted all available queries after {current_count} results.")
                    break
                continue

            q_node = random.choice(untried)
            comm_id = node_to_comm[q_node]
            gt_nodes = comm_to_nodes[comm_id]

            tried_queries.add((q_node, t_idx))

        else:
            part_len = len(full_part_seq[0]) if hasattr(full_part_seq[0], '__len__') else num_total_nodes
            safe_limit = min(curr_data.num_nodes, part_len)
            if safe_limit <= 0:
                continue
            deg_cache_key = (id(curr_data), safe_limit)
            if deg_cache_key not in _deg_cache:
                _deg_cache[deg_cache_key] = degree(curr_data.edge_index[0][:safe_limit], safe_limit)
            deg = _deg_cache[deg_cache_key]
            k_top = max(1, int(safe_limit * 0.2))
            top_nodes = torch.topk(deg, min(k_top, safe_limit)).indices.cpu().numpy()
            untried_top = [n for n in top_nodes if (int(n), t_idx) not in tried_queries]
            if not untried_top:
                _consecutive_empty += 1
                if _consecutive_empty > _max_consecutive_empty:
                    print(f"[Info] Exhausted all available queries after {current_count} results.")
                    break
                continue
            q_node = int(np.random.choice(untried_top))
            gt_nodes = set()

            tried_queries.add((q_node, t_idx))

        _consecutive_empty = 0
        m_idx = m_anchor_idx
        target_ts = full_gnn_seq[t_idx]['timestamp']
        m_ts = full_gnn_seq[m_idx]['timestamp']

        query_seeds = {q_node}
        if query_size > 1 and m_ts in ts_to_pyg_map:
            ref_ei = ts_to_pyg_map[m_ts].edge_index
            ref_ei = ref_ei[0] if isinstance(ref_ei, list) else ref_ei
            try:
                q_mask = ref_ei[0] == q_node
                neighbors = ref_ei[1][q_mask][:max(0, query_size - 1)].cpu().tolist()
                query_seeds.update(int(x) for x in neighbors)
            except Exception:
                pass

        candidate_nodes = None
        future_edge_index = None; future_num_nodes = num_total_nodes

        final_emb, final_sub, _, timings = run_combined_pipeline(
            diff_model, ref_model, all_snapshots_pyg, snapshots, diffusion_data,
            t_idx, m_idx, q_node, hops=3, device=device, dataset_name=dataset_name,
            diffusion_steps=diff_steps, num_total_nodes=num_total_nodes,
            pool_budget_sn=pool_budget_sn,
            candidate_nodes=candidate_nodes,
            future_edge_index=future_edge_index, future_num_nodes=future_num_nodes,
            max_out_pool=max_out
        )

        if final_emb is None:
            print("[!] Pipeline returned None embedding. Skipping...")
            continue

        t_s = time.time()
        max_c = max(3, min(len(final_sub), max_out))
        try:
            comm_nodes = set(sege_self_distillation(
                final_sub, final_emb, query_seeds,
                lambda_bal=0.5, alpha=alpha,
                min_community_size=3,
                max_community_size=max_c
            ))
        except Exception as _e:
            print("[SDR err]", _e)
            comm_nodes = set(query_seeds)
        comm_nodes &= set(final_sub.nodes())
        comm_nodes |= (set(query_seeds) & set(final_sub.nodes()))
        if q_node in final_sub.nodes():
            comm_nodes.add(q_node)
        if not comm_nodes:
            comm_nodes = {q_node}
        pred_graph = final_sub.subgraph(comm_nodes).copy()
        t_search_dur = time.time() - t_s

        comm_size = len(comm_nodes)
        current_count += 1

        pred_nodes = set(pred_graph.nodes())
        _fei = curr_data.edge_index
        _fei = _fei[0] if isinstance(_fei, list) else _fei
        true_graph_nx = nx.Graph()
        true_graph_nx.add_nodes_from(range(num_total_nodes))
        _fei_np = _fei.cpu().numpy()
        true_graph_nx.add_edges_from(zip(_fei_np[0].tolist(), _fei_np[1].tolist()))
        qual = evaluate_community_quality(pred_nodes, true_graph_nx)

        if has_ground_truth:
            gt_res = compute_set_metrics(pred_nodes, gt_nodes, set(final_sub.nodes()))
        else:
            gt_res = {"F1": 0.0, "Precision": 0.0, "Recall": 0.0, "Jaccard": 0.0, "NMI": 0.0}

        t_sketch = timings.get('Time_Region', 0)
        t_diff = timings.get('Time_Diffusion', 0)
        t_ref = timings.get('Time_Refinement', 0)

        metric_str = (
            f"Density={qual['Density']:.4f}, Conductance={qual['Conductance']:.4f}, Size={comm_size}"
        )
        if has_ground_truth:
            metric_str += f", Pre={gt_res['Precision']:.4f}, Recall={gt_res['Recall']:.4f}, F1={gt_res['F1']:.4f}"

        time_str = f"Sketch={t_sketch:.3f}s, Diff={t_diff:.3f}s, Ref={t_ref:.3f}s, Search={t_search_dur:.3f}s"
        print(f"[{current_count}/{target_count}] Q={q_node}: {metric_str} | {time_str}")

        results.append({
            "m_idx": m_idx, "t_idx": t_idx, "q_node": q_node,
            **gt_res,
            "Pred_Density": qual['Density'], "Pred_Conductance": qual['Conductance'],
            "Size": comm_size,
            "Time_Total": timings.get('Time_Total', 0) + t_search_dur
        })

        del final_emb, pred_graph, final_sub, comm_nodes, true_graph_nx
        if current_count % 5 == 0:
            gc.collect()

    if results:
        df = pd.DataFrame(results)

        print(f"\n{'='*60}")
        print(f"[Summary] Dataset: {dataset_name} | Queries: {len(df)} | mean / min / max")
        print(f"  Density:    {df['Pred_Density'].mean():.4f} / {df['Pred_Density'].min():.4f} / {df['Pred_Density'].max():.4f}")
        print(f"  Conductance:{df['Pred_Conductance'].mean():.4f} / {df['Pred_Conductance'].min():.4f} / {df['Pred_Conductance'].max():.4f}")
        if has_ground_truth:
            print(f"  Precision:  {df['Precision'].mean():.4f} / {df['Precision'].min():.4f} / {df['Precision'].max():.4f}")
            print(f"  Recall:     {df['Recall'].mean():.4f} / {df['Recall'].min():.4f} / {df['Recall'].max():.4f}")
            print(f"  F1-Score:   {df['F1'].mean():.4f} / {df['F1'].min():.4f} / {df['F1'].max():.4f}")
            print(f"  Jaccard:    {df['Jaccard'].mean():.4f} / {df['Jaccard'].min():.4f} / {df['Jaccard'].max():.4f}")
            print(f"  NMI:        {df['NMI'].mean():.4f} / {df['NMI'].min():.4f} / {df['NMI'].max():.4f}")
        print(f"{'='*60}")

        suffix = "gt" if has_ground_truth else "no_gt"
        save_path = f"./logs/{dataset_name}_node/final_results_{suffix}.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")
    else:
        print("[Warning] No valid queries were executed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='math')
    parser.add_argument('--diff_steps', type=int, default=20)
    parser.add_argument('--num_queries', type=int, default=1000)
    parser.add_argument('--query_size', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    P1 = f"./saved_models/{args.dataset}_phase1_data_splits.pkl"
    P2_C = f"./saved_models/{args.dataset}_phase2_embedding_splits_COARSE.pkl"
    MOD_DIFF = f"./saved_models/diffusion_model_v14_gnn_{args.dataset}.pth"
    MOD_REF = f"saved_models/refinement_gnn_v15_1_{args.dataset}.pth"

    main_multi_round_evaluation(
        MOD_DIFF, MOD_REF, DEVICE, args.dataset, P1, P2_C,
        args.diff_steps, args.num_queries, args.query_size, args.alpha
    )
