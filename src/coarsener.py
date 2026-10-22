import logging
from typing import Set, Dict, List, Tuple, Optional
from tqdm import tqdm
from coarse_graph import CoarseGraph
from cost_functions import ConvMatchCost, AConvMatchCost
from candidate_strategies import BaseCandidateStrategy, EmbeddingCandidateStrategy
import numpy as np
import heapq
import scipy.sparse as sp
import time
import torch
import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import faiss
    import faiss.contrib.torch_utils
    HAS_FAISS_TORCH = True
except ImportError:
    HAS_FAISS_TORCH = False
    logging.warning("[Perf Warning] faiss-gpu or torch interface not detected.")

try:
    import sparse_dot_mkl
    HAS_MKL = True
except ImportError:
    HAS_MKL = False
except Exception:
    HAS_MKL = False

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator

FP16_EPS = 1e-4
FP16_MAX = 65500.0
HUGE_GRAPH_THRESHOLD = 10_000_000
SEARCH_BATCH_SIZE = 50000
PRUNE_RATIO = 0.01
DECAY_FACTOR = 0.99

TARGET_MAX_DEGREE = 150.0
FULL_REBUILD_FREQ = 10

STRUCT_AFFINITY_WEIGHT = 0.15
STRUCT_SIZE_PENALTY_WEIGHT = 0.02

class GPUIndexManager:
    def __init__(self, d_dim: int, device_id: int = 0):
        self.d_dim = d_dim
        self.res = None
        self.index = None
        self.use_ivf = False
        self.fallback_to_cpu = False
        try:
            self.res = faiss.StandardGpuResources()
            self.res.setTempMemory(32 * 1024 * 1024)
        except Exception as e:
            logging.warning(f"[Faiss] GPU Resources init failed: {e}")
            self.res = None

    def train_and_add(self, data: np.ndarray):
        self.clean()
        n_samples = data.shape[0]
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=100.0, neginf=-100.0)
        if data.dtype != np.float32: data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data)

        if self.res is None:
            self._build_cpu_index(data)
            return

        force_ivf = n_samples > 15_000_000
        if not force_ivf:
            try:
                config = faiss.GpuIndexFlatConfig()
                config.device = 0
                self.index = faiss.GpuIndexFlatIP(self.res, self.d_dim, config)
                self.index.add(data)
                self.use_ivf = False
                return
            except RuntimeError:
                self.clean()

        try:
            nlist = min(int(4 * np.sqrt(n_samples)), 65536)
            if n_samples > 30_000_000: nlist = 32768
            quantizer = faiss.IndexFlatIP(self.d_dim)
            index_cpu = faiss.IndexIVFFlat(quantizer, self.d_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index_cpu.nprobe = min(6, nlist)
            gpu_index_ivf = faiss.index_cpu_to_gpu(self.res, 0, index_cpu)
            train_size = min(500000, n_samples)
            train_indices = np.random.choice(n_samples, train_size, replace=False)
            gpu_index_ivf.train(data[train_indices])
            batch_size = 1_000_000
            for i in range(0, n_samples, batch_size):
                gpu_index_ivf.add(data[i:i+batch_size])
            self.index = gpu_index_ivf
            self.use_ivf = True
            return
        except Exception as e:
            self.clean()
            self._build_cpu_index(data)

    def _build_cpu_index(self, data):
        self.fallback_to_cpu = True
        self.use_ivf = False
        self.index = faiss.IndexFlatIP(self.d_dim)
        self.index.add(data)

    def search_batched(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        n_query = query.shape[0]
        if query.dtype != np.float32: query = query.astype(np.float32)
        all_dists, all_inds = [], []
        current_batch_size = SEARCH_BATCH_SIZE
        if self.use_ivf: current_batch_size = current_batch_size // 2
        if self.fallback_to_cpu: current_batch_size = current_batch_size * 2

        for i in range(0, n_query, current_batch_size):
            end = min(i + current_batch_size, n_query)
            q_batch = query[i:end]
            try:
                D, I = self.index.search(q_batch, k)
            except RuntimeError:
                return np.array([]), np.array([])

            if isinstance(D, torch.Tensor):
                all_dists.append(D.cpu().numpy())
                all_inds.append(I.cpu().numpy())
            else:
                all_dists.append(D)
                all_inds.append(I)

        if not all_dists: return np.array([]), np.array([])
        return np.vstack(all_dists), np.vstack(all_inds)

    def clean(self):
        if self.index is not None:
            del self.index
            self.index = None
        self.fallback_to_cpu = False
        torch.cuda.empty_cache()

def _dynamic_pruning(adj: sp.csr_matrix,
                    decay: float = DECAY_FACTOR,
                    prune_ratio: float = PRUNE_RATIO,
                    target_max_degree: float = TARGET_MAX_DEGREE) -> sp.csr_matrix:
    if adj.nnz == 0: return adj

    n_nodes = adj.shape[0]
    nnz_per_row = adj.getnnz(axis=1)
    avg_degree = np.mean(nnz_per_row[nnz_per_row > 0]) if nnz_per_row.sum() > 0 else 0

    current_threshold_factor = prune_ratio

    if avg_degree > target_max_degree:
        overload_factor = avg_degree / target_max_degree
        current_threshold_factor = prune_ratio * (overload_factor ** 2.0)
        current_threshold_factor = min(current_threshold_factor, 0.5)

    if decay < 1.0:
        adj.data *= decay

    abs_data = np.abs(adj.data)
    avg_weight = np.mean(abs_data)
    threshold = avg_weight * current_threshold_factor

    mask = abs_data >= threshold

    if avg_degree < target_max_degree * 0.5 and np.sum(mask) < adj.nnz * 0.02:
         threshold *= 0.1
         mask = abs_data >= threshold

    adj.data[~mask] = 0
    adj.eliminate_zeros()
    return adj

@jit(nopython=True, cache=True)
def _fast_greedy_match(costs, sources, targets, num_nodes, max_merges):
    matched = np.zeros(num_nodes, dtype=np.bool_)
    merge_u = np.empty(max_merges, dtype=np.int32)
    merge_v = np.empty(max_merges, dtype=np.int32)
    count = 0
    n_candidates = len(costs)
    for i in range(n_candidates):
        if count >= max_merges: break
        u = sources[i]
        v = targets[i]
        if not matched[u] and not matched[v]:
            matched[u] = True; matched[v] = True
            merge_u[count] = u; merge_v[count] = v
            count += 1
    return merge_u[:count], merge_v[:count], matched

def _calibrate_merge_costs_for_community(adj_csr, sizes, pairs_u, pairs_v, costs):
    if costs.size == 0 or adj_csr is None:
        return costs
    if not sp.isspmatrix_csr(adj_csr):
        adj_csr = adj_csr.tocsr()

    pairs_u = np.asarray(pairs_u, dtype=np.int64)
    pairs_v = np.asarray(pairs_v, dtype=np.int64)
    costs = np.asarray(costs, dtype=np.float32)
    if pairs_u.size != costs.size or pairs_v.size != costs.size:
        return costs

    deg = np.asarray(adj_csr.sum(axis=1)).reshape(-1).astype(np.float32)
    sizes_np = np.asarray(sizes).reshape(-1).astype(np.float32)
    active_sizes = sizes_np[sizes_np > 0]
    mean_size = float(active_sizes.mean()) if active_sizes.size > 0 else 1.0

    edge_w = np.asarray(adj_csr[pairs_u, pairs_v]).reshape(-1).astype(np.float32)
    affinity = edge_w / np.sqrt((deg[pairs_u] + FP16_EPS) * (deg[pairs_v] + FP16_EPS))
    affinity = np.clip(affinity, 0.0, 1.0)
    merged_size_ratio = (sizes_np[pairs_u] + sizes_np[pairs_v]) / max(mean_size, FP16_EPS)
    size_penalty = np.log1p(np.maximum(merged_size_ratio - 2.0, 0.0)).astype(np.float32)

    calibrated = costs * (1.0 - STRUCT_AFFINITY_WEIGHT * affinity)
    calibrated += STRUCT_SIZE_PENALTY_WEIGHT * size_penalty
    return calibrated.astype(np.float32, copy=False)

def _scipy_to_torch_sparse(adj_scipy: sp.csr_matrix, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    coo = adj_scipy.tocoo()
    row = torch.from_numpy(coo.row).long().to(device)
    col = torch.from_numpy(coo.col).long().to(device)
    values = torch.from_numpy(coo.data).to(dtype=dtype, device=device)
    indices = torch.stack([row, col], dim=0)
    shape = coo.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=device).coalesce()

def _safe_fp16_tensor(data_np, device):
    return torch.from_numpy(data_np).half().to(device)

def _safe_fp32_tensor(data_np, device):
    return torch.from_numpy(data_np).float().to(device)

def _batched_spmm_to_cpu(adj_csr: sp.csr_matrix, features_gpu: torch.Tensor, batch_size: int = 500000, device: torch.device = None) -> torch.Tensor:
    N = adj_csr.shape[0]
    out_dim = features_gpu.shape[1]
    result_cpu = torch.zeros((N, out_dim), dtype=torch.float16)
    X = features_gpu
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        adj_batch = adj_csr[i:end, :]
        if adj_batch.nnz == 0: continue
        adj_batch_coo = adj_batch.tocoo()
        row = torch.from_numpy(adj_batch_coo.row).long().to(device)
        col = torch.from_numpy(adj_batch_coo.col).long().to(device)
        values = torch.from_numpy(adj_batch_coo.data).float().to(device)
        indices = torch.stack([row, col], dim=0)
        adj_torch = torch.sparse_coo_tensor(indices, values, torch.Size(adj_batch.shape), device=device).coalesce()
        res_batch = torch.sparse.mm(adj_torch, X.float())
        res_batch = torch.nan_to_num(res_batch, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
        result_cpu[i:end] = res_batch.half().cpu()
        del adj_torch, row, col, values, indices, res_batch, adj_batch
        if i % (batch_size * 5) == 0: torch.cuda.empty_cache()
    return result_cpu

def static_coarsen(
    initial_adj: sp.csr_matrix,
    initial_features: np.ndarray,
    candidate_strategy,
    cost_function,
    target_coarsening_ratio: float
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_orig = initial_adj.shape[0]

    use_huge_graph_mode = N_orig > HUGE_GRAPH_THRESHOLD

    target_num_supernodes = int(N_orig * target_coarsening_ratio)
    current_num_supernodes = N_orig
    level = 0
    node_mapping = np.arange(N_orig, dtype=np.int32)

    frozen_adj_cpu = initial_adj if sp.isspmatrix_csr(initial_adj) else initial_adj.tocsr()
    batch_mapper = np.arange(N_orig, dtype=np.int32)
    frozen_num_nodes = N_orig

    try:
        current_features_gpu = _safe_fp16_tensor(initial_features, device)
        current_sizes_gpu = torch.ones(N_orig, dtype=torch.float32, device=device)
    except RuntimeError:
        logging.error("Insufficient GPU memory to store the initial feature matrix.")
        raise

    total_nodes_to_reduce = N_orig - target_num_supernodes
    pbar = tqdm(total=total_nodes_to_reduce, desc="Static Coarsening", unit="merges")

    current_k = candidate_strategy.k_neighbors
    GCN_UPDATE_FREQ = 5

    h_tilde_store = None
    gpu_faiss_manager = GPUIndexManager(d_dim=initial_features.shape[1])

    while current_num_supernodes > target_num_supernodes:
        level += 1
        if level % 5 == 0: torch.cuda.empty_cache()

        is_refresh_step = (level == 1) or (level % GCN_UPDATE_FREQ == 0)

        if is_refresh_step:
            if frozen_num_nodes != current_num_supernodes:
                rows, cols = frozen_adj_cpu.nonzero()
                new_rows = batch_mapper[rows]
                new_cols = batch_mapper[cols]
                new_data = frozen_adj_cpu.data
                frozen_adj_cpu = sp.csr_matrix(
                    (new_data, (new_rows, new_cols)),
                    shape=(current_num_supernodes, current_num_supernodes)
                )
                frozen_adj_cpu = _dynamic_pruning(frozen_adj_cpu)
                frozen_num_nodes = current_num_supernodes
                batch_mapper = np.arange(current_num_supernodes, dtype=np.int32)

            d_vec_cpu = np.array(frozen_adj_cpu.sum(axis=1)).flatten()
            d_vec_gpu = torch.from_numpy(d_vec_cpu).float().to(device)
            d_vec_gpu.add_(current_sizes_gpu).add_(FP16_EPS).pow_(-0.5)
            d_vec_gpu = torch.nan_to_num(d_vec_gpu, nan=0.0, posinf=0.0)

            X_float = current_features_gpu.float()
            X_scaled = X_float * d_vec_gpu.unsqueeze(1)

            if use_huge_graph_mode:
                h_tilde_base = _batched_spmm_to_cpu(frozen_adj_cpu, X_scaled.half(), batch_size=200000, device=device)
            else:
                current_adj_gpu = _scipy_to_torch_sparse(frozen_adj_cpu, device)
                h_tilde_base = torch.sparse.mm(current_adj_gpu, X_scaled)
                del current_adj_gpu

            if use_huge_graph_mode:
                h_tilde_store = h_tilde_base.float()
                X_scaled_cpu = X_scaled.cpu()
                sizes_cpu = current_sizes_gpu.cpu()
                d_vec_cpu = d_vec_gpu.cpu()
                term2 = X_scaled_cpu * sizes_cpu.unsqueeze(1)
                h_tilde_store.add_(term2)
                h_tilde_store.mul_(d_vec_cpu.unsqueeze(1))
                h_tilde_store = h_tilde_store.half()
                del X_scaled_cpu, sizes_cpu, d_vec_cpu, h_tilde_base
            else:
                term2 = X_scaled * current_sizes_gpu.unsqueeze(1)
                h_tilde_base.add_(term2)
                h_tilde_base.mul_(d_vec_gpu.unsqueeze(1))
                h_tilde_base = torch.nan_to_num(h_tilde_base, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
                h_tilde_store = h_tilde_base.half()

            del d_vec_gpu, X_float, X_scaled

        if use_huge_graph_mode or not h_tilde_store.is_cuda:
            h_search = h_tilde_store.float().numpy()
            faiss.normalize_L2(h_search)
        else:
            h_search_gpu = h_tilde_store.float()
            h_search_gpu = torch.nn.functional.normalize(h_search_gpu, p=2, dim=1)
            h_search = h_search_gpu.cpu().numpy()

        torch.cuda.empty_cache()
        gpu_faiss_manager.train_and_add(h_search)
        k_search = current_k + 1
        _, I_faiss_np = gpu_faiss_manager.search_batched(h_search, k_search)
        if I_faiss_np.size == 0: break
        I_faiss = torch.from_numpy(I_faiss_np)
        gpu_faiss_manager.clean()
        del h_search

        results_u, results_v, results_c = [], [], []
        chunk_size = 500000
        num_rows = I_faiss.shape[0]
        h_tilde_is_cpu = not h_tilde_store.is_cuda

        for start_idx in range(0, num_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, num_rows)
            targets_chunk = I_faiss[start_idx:end_idx].reshape(-1)
            sources_chunk = torch.arange(start_idx, end_idx, dtype=torch.long).repeat_interleave(k_search)
            mask = (targets_chunk != -1) & (targets_chunk > sources_chunk) & (targets_chunk < current_num_supernodes)
            if not mask.any(): continue
            batch_u = sources_chunk[mask]
            batch_v = targets_chunk[mask]
            batch_u_gpu = batch_u.to(device)
            batch_v_gpu = batch_v.to(device)

            if h_tilde_is_cpu:
                unique_nodes = torch.cat([batch_u, batch_v]).unique()
                unique_nodes_gpu = unique_nodes.to(device)
                temp_map = torch.full((current_num_supernodes,), -1, dtype=torch.long, device=device)
                temp_map[unique_nodes_gpu] = torch.arange(len(unique_nodes), device=device)
                local_u = temp_map[batch_u_gpu]
                local_v = temp_map[batch_v_gpu]
                sub_h_tilde = h_tilde_store[unique_nodes].to(device)
                sub_feats = current_features_gpu[unique_nodes_gpu]
                sub_sizes = current_sizes_gpu[unique_nodes_gpu]
                batch_pairs = torch.stack([local_u, local_v], dim=1)
                batch_costs = cost_function.calculate_batch_torch(
                    features=sub_feats, d_tilde_vec=sub_sizes, sizes=sub_sizes,
                    h_tilde=sub_h_tilde, pairs=batch_pairs
                )
                del sub_h_tilde, sub_feats, sub_sizes, unique_nodes_gpu, temp_map
            else:
                batch_pairs = torch.stack([batch_u_gpu, batch_v_gpu], dim=1)
                batch_costs = cost_function.calculate_batch_torch(
                    features=current_features_gpu, d_tilde_vec=current_sizes_gpu,
                    sizes=current_sizes_gpu, h_tilde=h_tilde_store, pairs=batch_pairs
                )
            results_u.append(batch_u.numpy())
            results_v.append(batch_v.numpy())
            results_c.append(batch_costs.float().cpu().numpy())

        del I_faiss
        torch.cuda.empty_cache()

        merges_u, merges_v, matched_nodes = np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.zeros(current_num_supernodes, dtype=bool)

        if len(results_u) > 0:
            cpu_u, cpu_v = np.concatenate(results_u), np.concatenate(results_v)
            cpu_costs = np.concatenate(results_c)
            cpu_costs = _calibrate_merge_costs_for_community(
                frozen_adj_cpu, current_sizes_gpu.cpu().numpy(), cpu_u, cpu_v, cpu_costs
            )

            sort_idx = np.argsort(cpu_costs)
            cpu_u, cpu_v, cpu_costs = cpu_u[sort_idx], cpu_v[sort_idx], cpu_costs[sort_idx]
            del sort_idx

            max_merges = (current_num_supernodes - target_num_supernodes)
            if HAS_NUMBA:
                merges_u, merges_v, matched_nodes = _fast_greedy_match(cpu_costs, cpu_u, cpu_v, current_num_supernodes, max_merges)
            else:
                matched = np.zeros(current_num_supernodes, dtype=bool)
                u_list, v_list = [], []
                cnt = 0
                for i in range(len(cpu_u)):
                    u, v = cpu_u[i], cpu_v[i]
                    if not matched[u] and not matched[v]:
                        matched[u] = matched[v] = True
                        u_list.append(u); v_list.append(v)
                        cnt += 1
                        if cnt >= max_merges: break
                merges_u = np.array(u_list, dtype=np.int32)
                merges_v = np.array(v_list, dtype=np.int32)
                matched_nodes = matched

        num_merges = len(merges_u)
        if num_merges == 0: break
        efficiency = num_merges / current_num_supernodes
        if efficiency < 0.005: break

        pbar.update(num_merges)
        pbar.set_postfix({"nodes": f"{current_num_supernodes-num_merges}"})

        step_map = np.full(current_num_supernodes, -1, dtype=np.int32)
        new_ids_merged = np.arange(num_merges, dtype=np.int32)
        step_map[merges_u] = new_ids_merged
        step_map[merges_v] = new_ids_merged
        unmatched_mask = ~matched_nodes
        num_unmatched = np.sum(unmatched_mask)
        step_map[unmatched_mask] = np.arange(num_merges, num_merges + num_unmatched, dtype=np.int32)

        next_num_supernodes = num_merges + num_unmatched
        batch_mapper = step_map[batch_mapper]
        node_mapping = step_map[node_mapping]

        mapping_gpu = torch.from_numpy(step_map).long().to(device)
        next_sizes_gpu = torch.zeros(next_num_supernodes, dtype=torch.float32, device=device)
        next_sizes_gpu.index_add_(0, mapping_gpu, current_sizes_gpu)
        inv_sizes_gpu = torch.zeros_like(next_sizes_gpu)
        mask_nz = next_sizes_gpu > 0
        inv_sizes_gpu[mask_nz] = 1.0 / next_sizes_gpu[mask_nz]
        inv_sizes_unsq = inv_sizes_gpu.unsqueeze(1)
        weighted_features_gpu = current_features_gpu.float() * current_sizes_gpu.unsqueeze(1)
        next_features_sum_gpu = torch.zeros(next_num_supernodes, weighted_features_gpu.size(1),
                                            dtype=torch.float32, device=device)
        next_features_sum_gpu.index_add_(0, mapping_gpu, weighted_features_gpu)
        next_features_gpu = next_features_sum_gpu * inv_sizes_unsq
        next_features_gpu = torch.nan_to_num(next_features_gpu, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
        next_features_gpu_half = next_features_gpu.half()

        current_features_gpu = next_features_gpu_half
        current_sizes_gpu = next_sizes_gpu
        current_num_supernodes = next_num_supernodes

        del weighted_features_gpu, next_features_sum_gpu, inv_sizes_gpu, inv_sizes_unsq
        del next_features_gpu, next_features_gpu_half

    pbar.close()

    if frozen_num_nodes != current_num_supernodes:
        rows, cols = frozen_adj_cpu.nonzero()
        new_rows = batch_mapper[rows]
        new_cols = batch_mapper[cols]
        new_data = frozen_adj_cpu.data
        frozen_adj_cpu = sp.csr_matrix((new_data, (new_rows, new_cols)), shape=(current_num_supernodes, current_num_supernodes))
        frozen_adj_cpu = _dynamic_pruning(frozen_adj_cpu)

    final_adj_full = frozen_adj_cpu
    final_features_full = current_features_gpu.float().cpu().numpy()
    final_sizes_full = current_sizes_gpu.cpu().numpy().astype(int)

    final_graph = object.__new__(CoarseGraph)
    final_graph.num_original_nodes = N_orig
    final_graph.current_base_adj = initial_adj.tocsr()
    final_graph.original_features = initial_features
    final_graph.adj = final_adj_full
    final_graph.features = final_features_full
    final_graph.supernode_sizes = final_sizes_full
    final_graph.node_to_supernode = node_mapping
    final_graph.degree = np.array(final_adj_full.sum(axis=1)).flatten()
    sorted_indices = np.argsort(node_mapping)
    sorted_sids = node_mapping[sorted_indices]
    unique_sids, idx_start = np.unique(sorted_sids, return_index=True)
    final_supernode_to_nodes_map = {}
    for i, sid in enumerate(unique_sids):
        start = idx_start[i]
        end = idx_start[i+1] if i < len(unique_sids) - 1 else len(sorted_sids)
        final_supernode_to_nodes_map[sid] = sorted_indices[start:end].tolist()
    final_graph.supernode_to_nodes = final_supernode_to_nodes_map

    return final_graph

class IncrementalCoarsener:
    def __init__(self,
                 initial_coarse_graph: CoarseGraph,
                 cost_function: ConvMatchCost,
                 candidate_strategy: BaseCandidateStrategy,
                 merge_threshold: float = 50.0,
                 proactive_merge_k: int = 0,
                 split_threshold: int = 100):

        if not sp.isspmatrix_csr(initial_coarse_graph.adj):
            initial_coarse_graph.adj = initial_coarse_graph.adj.tocsr()

        self.coarse_graph = initial_coarse_graph
        self.cost_function = cost_function
        self.candidate_strategy = candidate_strategy
        self.merge_threshold = merge_threshold
        self.proactive_merge_k = proactive_merge_k
        self.split_threshold = split_threshold

        self.initial_num_supernodes = initial_coarse_graph.adj.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_target_ratio = 0.95

        self.snapshot_counter = 0

        self.features_gpu = _safe_fp16_tensor(initial_coarse_graph.features, self.device)
        self.sizes_gpu = _safe_fp32_tensor(initial_coarse_graph.supernode_sizes, self.device)
        self.gpu_faiss_manager = GPUIndexManager(d_dim=self.features_gpu.shape[1])

        self.coarse_graph.adj = _dynamic_pruning(self.coarse_graph.adj)

    def set_target_ratio(self, ratio: float):
        self.current_target_ratio = ratio

    def _compute_gcn_gpu_inplace(self, adj_scipy):
        N = adj_scipy.shape[0]

        d_vec_cpu = np.array(adj_scipy.sum(axis=1)).flatten()

        d_vec = torch.from_numpy(d_vec_cpu).float().to(self.device)
        d_vec.add_(self.sizes_gpu).add_(FP16_EPS).pow_(-0.5)
        d_vec = torch.nan_to_num(d_vec, nan=0.0, posinf=0.0)

        X_float = self.features_gpu.float()
        X_scaled = X_float * d_vec.unsqueeze(1)

        if N > HUGE_GRAPH_THRESHOLD:
            h_tilde_base = _batched_spmm_to_cpu(adj_scipy, X_scaled.half(), device=self.device)
            h_tilde = h_tilde_base.to(self.device).float()
            del h_tilde_base
        else:
            adj_gpu = _scipy_to_torch_sparse(adj_scipy, self.device, dtype=torch.float32)
            h_tilde = torch.sparse.mm(adj_gpu, X_scaled)
            del adj_gpu

        term2 = X_scaled * self.sizes_gpu.unsqueeze(1)
        h_tilde.add_(term2)
        h_tilde.mul_(d_vec.unsqueeze(1))

        h_tilde = torch.nan_to_num(h_tilde, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
        return h_tilde.half()

    def _periodic_full_rebuild(self):
        self.coarse_graph.adj = _dynamic_pruning(
            self.coarse_graph.adj,
            decay=DECAY_FACTOR,
            prune_ratio=PRUNE_RATIO * 1.5,
            target_max_degree=TARGET_MAX_DEGREE * 0.8
        )

        torch.cuda.empty_cache()
        gc.collect()

    def apply_changes_batch(self, added_edges: set, deleted_edges: set):
        self.snapshot_counter += 1

        if self.snapshot_counter % FULL_REBUILD_FREQ == 0:
            self._periodic_full_rebuild()

        graph = self.coarse_graph
        current_nnz = graph.adj.nnz
        current_num = graph.adj.shape[0]
        current_density = current_nnz / max(1, current_num)

        SKIP_MERGE = False
        if current_density > TARGET_MAX_DEGREE * 2.0:
             SKIP_MERGE = True

        if not added_edges and not deleted_edges:
            if not SKIP_MERGE and self.proactive_merge_k > 0 and current_density < TARGET_MAX_DEGREE:
                 if not sp.isspmatrix_csr(graph.adj): graph.adj = graph.adj.tocsr()
                 try:
                     h_tilde_lazy = self._compute_gcn_gpu_inplace(graph.adj)
                     self._proactive_merge_loop(h_tilde_lazy, current_density)
                     del h_tilde_lazy
                 except RuntimeError: pass
            return

        graph.update_base_adj(added_edges, deleted_edges)

        batch_affected_nodes = set()
        for u, v in added_edges: batch_affected_nodes.update([u, v])
        for u, v in deleted_edges: batch_affected_nodes.update([u, v])
        node_list_to_process = sorted(list(batch_affected_nodes))

        split_nodes_info = []
        affected_sids_set = set()

        if node_list_to_process:
            if len(node_list_to_process) < self.split_threshold:
                for node_id in node_list_to_process:
                    split_result = self.coarse_graph.split_supernode(node_id)
                    if split_result is None:
                        current_supernode_id = self.coarse_graph.get_supernode_id(node_id)
                        split_nodes_info.append((current_supernode_id, -1))
                        affected_sids_set.add(current_supernode_id)
                    else:
                        original_sid, new_sid = split_result
                        split_nodes_info.append((new_sid, original_sid))
                        affected_sids_set.add(original_sid); affected_sids_set.add(new_sid)
            else:
                original_sid_map = {nid: self.coarse_graph.get_supernode_id(nid) for nid in node_list_to_process}
                affected_sids_set = self.coarse_graph.split_supernodes_batch_CSR(node_list_to_process)
                for nid in node_list_to_process:
                    new_sid = self.coarse_graph.get_supernode_id(nid)
                    original_sid = original_sid_map.get(nid, -1)
                    split_nodes_info.append((new_sid, original_sid))

        if SKIP_MERGE:
            if self.snapshot_counter % 5 == 0:
                logging.warning(f"  [Overload] Density {current_density:.1f} > Limit. Skipping merges.")
            return

        if affected_sids_set:
            self.features_gpu = _safe_fp16_tensor(graph.features, self.device)
            self.sizes_gpu = _safe_fp32_tensor(graph.supernode_sizes, self.device)

        if not sp.isspmatrix_csr(graph.adj): graph.adj = graph.adj.tocsr()

        try:
            h_tilde_shared = self._compute_gcn_gpu_inplace(graph.adj)
        except RuntimeError:
            torch.cuda.empty_cache()
            return
        except Exception as e:
            logging.warning(f"GCN Error: {e}, skipping merge step.")
            return

        if split_nodes_info:
            self._reactive_merge_step(split_nodes_info, h_tilde_shared, current_density)

        if self.proactive_merge_k > 0 or len(affected_sids_set) > 0:
            self._proactive_merge_loop(h_tilde_shared, current_density)

        del h_tilde_shared
        torch.cuda.empty_cache()

    def _reactive_merge_step(self, split_nodes_info, h_tilde_gpu, current_density):
        graph = self.coarse_graph
        query_sids = [sid for (sid, _) in split_nodes_info if graph.supernode_sizes[sid] > 0]
        if not query_sids: return

        k_neigh = 2

        active_indices = torch.nonzero(self.sizes_gpu > 0).reshape(-1)
        if active_indices.numel() == 0: return

        h_search_active = h_tilde_gpu.index_select(0, active_indices).float()
        h_search_active = torch.nn.functional.normalize(h_search_active, p=2, dim=1)
        h_search_active_np = h_search_active.cpu().numpy()

        self.gpu_faiss_manager.train_and_add(h_search_active_np)

        q_tensor = h_tilde_gpu[query_sids].float()
        q_tensor = torch.nn.functional.normalize(q_tensor, p=2, dim=1)
        q_np = q_tensor.cpu().numpy()

        _, I_local = self.gpu_faiss_manager.search_batched(q_np, k_neigh + 1)
        self.gpu_faiss_manager.clean()

        active_indices_cpu = active_indices.cpu().numpy()
        node_to_original = {sid: orig_sid for (sid, orig_sid) in split_nodes_info}
        pairs_to_process = []
        added_pairs = set()

        for i, q_sid in enumerate(query_sids):
            orig_sid = node_to_original.get(q_sid, -1)
            candidates = []
            for local_idx in I_local[i]:
                if local_idx != -1 and local_idx < len(active_indices_cpu):
                    candidates.append(active_indices_cpu[local_idx])

            if orig_sid != -1 and orig_sid not in candidates and graph.supernode_sizes[orig_sid] > 0:
                candidates.append(orig_sid)

            for cand_sid in candidates:
                if cand_sid == -1 or cand_sid == q_sid: continue
                if graph.supernode_sizes[cand_sid] == 0: continue
                pair = tuple(sorted((q_sid, cand_sid)))
                if pair not in added_pairs:
                    pairs_to_process.append(pair)
                    added_pairs.add(pair)

        if not pairs_to_process: return

        costs_array = self._calculate_batch_costs_torch_direct(
            pairs_to_process, self.features_gpu, self.sizes_gpu, h_tilde_gpu, self.sizes_gpu
        )
        if costs_array.size > 0:
            pair_u = np.fromiter((p[0] for p in pairs_to_process), dtype=np.int64, count=len(pairs_to_process))
            pair_v = np.fromiter((p[1] for p in pairs_to_process), dtype=np.int64, count=len(pairs_to_process))
            costs_array = _calibrate_merge_costs_for_community(
                graph.adj, graph.supernode_sizes, pair_u, pair_v, costs_array
            )

        current_threshold = self.merge_threshold
        best_costs = {}
        for i, (u, v) in enumerate(pairs_to_process):
            if i >= len(costs_array): break
            cost = costs_array[i]
            if u in query_sids:
                if u not in best_costs or cost < best_costs[u][0]: best_costs[u] = (cost, v)
            if v in query_sids:
                if v not in best_costs or cost < best_costs[v][0]: best_costs[v] = (cost, u)

        pq = []
        for u, (c, v) in best_costs.items(): heapq.heappush(pq, (c, u, v))

        merged_nodes = set()
        merge_map = {}
        while pq:
            c, u, v = heapq.heappop(pq)
            if u in merged_nodes or v in merged_nodes or v == -1: continue
            if c < current_threshold:
                if graph.supernode_sizes[v] > 0:
                    merge_map[u] = v
                    merged_nodes.add(u); merged_nodes.add(v)

        if merge_map:
            self._apply_merge_map_gpu(merge_map)

    def _proactive_merge_loop(self, h_tilde_gpu, current_density):
        if current_density > TARGET_MAX_DEGREE * 1.5:
            return

        N_current = self.coarse_graph.adj.shape[0]

        graph = self.coarse_graph
        current_threshold = self.merge_threshold

        while True:
            active_supernodes = np.where(graph.supernode_sizes > 0)[0]
            if len(active_supernodes) < 2: break

            target_k = 20
            if current_density > 100.0: target_k = 10
            if current_density < 2.0 and N_current > self.initial_num_supernodes * 0.8: target_k = 50

            k_proactive = min(target_k, len(active_supernodes))

            if k_proactive <= 0: break

            supernodes_to_process = np.random.choice(active_supernodes, k_proactive, replace=False)

            active_indices_th = torch.from_numpy(active_supernodes).to(self.device)
            h_search_active = h_tilde_gpu.index_select(0, active_indices_th).float()
            h_search_active = torch.nn.functional.normalize(h_search_active, p=2, dim=1)
            h_search_active_np = h_search_active.cpu().numpy()

            k_neigh = 2
            self.gpu_faiss_manager.train_and_add(h_search_active_np)

            q_tensor = h_tilde_gpu[supernodes_to_process].float()
            q_tensor = torch.nn.functional.normalize(q_tensor, p=2, dim=1)
            q_np = q_tensor.cpu().numpy()

            _, I_local = self.gpu_faiss_manager.search_batched(q_np, k_neigh + 1)
            self.gpu_faiss_manager.clean()

            pairs_to_process = []
            added_pairs = set()
            for i, u in enumerate(supernodes_to_process):
                for local_v_idx in I_local[i]:
                    if local_v_idx == -1 or local_v_idx >= len(active_supernodes): continue
                    v = active_supernodes[local_v_idx]
                    if v == u or graph.supernode_sizes[v] == 0: continue
                    pair = tuple(sorted((u, v)))
                    if pair not in added_pairs:
                        pairs_to_process.append(pair)
                        added_pairs.add(pair)

            if not pairs_to_process: break

            costs_array = self._calculate_batch_costs_torch_direct(
                pairs_to_process, self.features_gpu, self.sizes_gpu, h_tilde_gpu, self.sizes_gpu
            )
            if costs_array.size > 0:
                pair_u = np.fromiter((p[0] for p in pairs_to_process), dtype=np.int64, count=len(pairs_to_process))
                pair_v = np.fromiter((p[1] for p in pairs_to_process), dtype=np.int64, count=len(pairs_to_process))
                costs_array = _calibrate_merge_costs_for_community(
                    graph.adj, graph.supernode_sizes, pair_u, pair_v, costs_array
                )

            merges_to_collect = []
            valid_merge_count = 0
            collect_threshold = current_threshold * 1.2

            for i, (u, v) in enumerate(pairs_to_process):
                if i < len(costs_array) and costs_array[i] < collect_threshold:
                    heapq.heappush(merges_to_collect, (costs_array[i], u, v))
                    valid_merge_count += 1

            if valid_merge_count < len(pairs_to_process) * 0.005: break

            merged_this_step = set()
            merge_map = {}
            while merges_to_collect:
                _, u, v = heapq.heappop(merges_to_collect)
                if u not in merged_this_step and v not in merged_this_step:
                    merge_map[v] = u
                    merged_this_step.add(u); merged_this_step.add(v)

            if not merge_map: break
            self._apply_merge_map_gpu(merge_map)

            if len(merge_map) < 5: break

    def _calculate_batch_costs_torch_direct(self, pairs, feat_th, sizes_th, h_th, d_vec_th):
        if not hasattr(self.cost_function, 'calculate_batch_torch'): return np.array([])
        num_pairs = len(pairs)
        pairs_cpu = torch.tensor(pairs, dtype=torch.long)
        all_costs = []
        bs = 100000
        with torch.no_grad():
            for i in range(0, num_pairs, bs):
                batch_pairs = pairs_cpu[i:min(i+bs, num_pairs)]
                batch_pairs_gpu = batch_pairs.to(self.device)
                costs = self.cost_function.calculate_batch_torch(
                    feat_th, d_vec_th, sizes_th, h_th, batch_pairs_gpu
                )
                all_costs.append(costs.float().cpu().numpy())
                del batch_pairs_gpu, costs
        return np.concatenate(all_costs) if all_costs else np.array([])

    def _apply_merge_map_gpu(self, merge_map):
        graph = self.coarse_graph
        N = graph.adj.shape[0]

        P_batch = sp.identity(N, format='lil', dtype=graph.adj.dtype)
        for c, p in merge_map.items():
            P_batch[c, c] = 0
            P_batch[c, p] = 1
        P_batch = P_batch.tocsr()

        if HAS_MKL:
            graph.adj = sparse_dot_mkl.dot_product_mkl(P_batch.T, graph.adj, cast=True)
            graph.adj = sparse_dot_mkl.dot_product_mkl(graph.adj, P_batch, cast=True)
        else:
            graph.adj = P_batch.T @ graph.adj @ P_batch

        graph.adj = _dynamic_pruning(
            graph.adj,
            decay=DECAY_FACTOR,
            prune_ratio=PRUNE_RATIO,
            target_max_degree=TARGET_MAX_DEGREE
        )

        graph.degree = np.array(graph.adj.sum(axis=1)).flatten()

        with torch.no_grad():
            srcs = torch.tensor(list(merge_map.keys()), device=self.device, dtype=torch.long)
            dsts = torch.tensor(list(merge_map.values()), device=self.device, dtype=torch.long)

            c_sizes = self.sizes_gpu.index_select(0, srcs)
            c_feats = self.features_gpu.index_select(0, srcs).float() * c_sizes.unsqueeze(1)

            add_size = torch.zeros_like(self.sizes_gpu)
            add_size.index_add_(0, dsts, c_sizes)

            feat_buffer = torch.zeros(self.features_gpu.shape[0], self.features_gpu.shape[1],
                                      dtype=torch.float32, device=self.device)
            feat_buffer.index_add_(0, dsts, c_feats)

            uniq_dsts = torch.unique(dsts)
            p_old_sz = self.sizes_gpu.index_select(0, uniq_dsts)
            p_old_ft = self.features_gpu.index_select(0, uniq_dsts).float()

            p_add_sz = add_size.index_select(0, uniq_dsts)
            p_add_ft = feat_buffer.index_select(0, uniq_dsts)

            p_new_sz = p_old_sz + p_add_sz
            p_new_sz_safe = p_new_sz.unsqueeze(1).clone()
            p_new_sz_safe[p_new_sz_safe == 0] = 1.0

            p_new_ft = (p_old_ft * p_old_sz.unsqueeze(1) + p_add_ft) / p_new_sz_safe

            self.sizes_gpu.index_copy_(0, uniq_dsts, p_new_sz)
            p_new_ft = torch.nan_to_num(p_new_ft, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
            self.features_gpu.index_copy_(0, uniq_dsts, p_new_ft.half())

            self.sizes_gpu.index_fill_(0, srcs, 0.0)
            self.features_gpu.index_fill_(0, srcs, 0.0)

            affected = torch.cat([srcs, uniq_dsts])
            aff_cpu = affected.cpu().numpy()

            f_sub = self.features_gpu.index_select(0, affected).float().cpu().numpy()
            s_sub = self.sizes_gpu.index_select(0, affected).cpu().numpy()

            graph.features[aff_cpu] = f_sub
            graph.supernode_sizes[aff_cpu] = s_sub.astype(graph.supernode_sizes.dtype)

            del feat_buffer, c_feats, p_new_ft

        for c, p in merge_map.items():
             graph.node_to_supernode[graph.node_to_supernode == c] = p

        for c, p in merge_map.items():
            nodes = graph.supernode_to_nodes.pop(c, [])
            if p not in graph.supernode_to_nodes: graph.supernode_to_nodes[p] = []
            graph.supernode_to_nodes[p].extend(nodes)

    def get_coarse_graph(self) -> CoarseGraph:
        if not sp.isspmatrix_csr(self.coarse_graph.adj):
            self.coarse_graph.adj = self.coarse_graph.adj.tocsr()
        return self.coarse_graph
