import torch
import pickle
import numpy as np
import os
from torch.utils.data import Dataset
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph, k_hop_subgraph
import joblib
from tqdm import tqdm
import gc

class TemporalEmbeddingDataset(Dataset):
    def __init__(self,
                 p1_pkl_path: str,
                 p2_pkl_path: str,
                 dataset_name: str,
                 split: str,
                 min_history_len=2,
                 max_history_len=5,
                 samples_per_snapshot: int = 2000,
                 dsm_tau: float = 0.8,
                 forecast_horizons=(1,),
                 max_train_samples: int = 50000,
                 max_val_samples: int = 5000,
                 preloaded_inputs=None,
                 preloaded_p2_raw=None,
                 preloaded_num_nodes=None
                ):

        if split == 'train':
            self.max_samples = max_train_samples
        else:
            self.max_samples = max_val_samples

        print(f"--- Loading '{split}' split data for Diffusion (V14.7 RAM Optimized) ---")

        self.dataset_name = dataset_name
        self.split = split
        self.dsm_tau = dsm_tau
        self.forecast_horizons = tuple(sorted(set(int(h) for h in forecast_horizons)))
        if not self.forecast_horizons or self.forecast_horizons[0] < 1:
            raise ValueError("forecast_horizons must contain positive integers")
        self.subgraph_size_limit = samples_per_snapshot

        if preloaded_inputs is not None and preloaded_num_nodes is not None:
            print("  - [Memory Opt] Preloaded data detected, skipping P1 file read.")
            self.num_nodes_global = preloaded_num_nodes
            self.aligned_inputs = preloaded_inputs

            self.snapshot_node_counts = [inp['adj'].shape[0] for inp in self.aligned_inputs]
            self.snapshot_degrees = []
            self.snapshot_num_edges = []

            for inp in self.aligned_inputs:
                adj = inp['adj']
                deg = np.array(adj.sum(axis=1)).flatten()
                self.snapshot_degrees.append(torch.tensor(deg, dtype=torch.float))
                self.snapshot_num_edges.append(adj.nnz // 2)

        else:
            print(f"  - Loading P1 (Layout & Adj) via Joblib: {p1_pkl_path}")
            try:
                phase1_data = joblib.load(p1_pkl_path)
            except Exception:
                with open(p1_pkl_path, 'rb') as f:
                    phase1_data = pickle.load(f)

            self.num_nodes_global = phase1_data['num_total_nodes']

            try:
                self.aligned_inputs = phase1_data[f'{split}_aligned_inputs']
                del phase1_data
                gc.collect()
                print("  - P1 raw data cleared, releasing memory.")

                self.snapshot_node_counts = [inp['adj'].shape[0] for inp in self.aligned_inputs]
                self.snapshot_degrees = []
                self.snapshot_num_edges = []

                for inp in self.aligned_inputs:
                    adj = inp['adj']
                    deg = np.array(adj.sum(axis=1)).flatten()
                    self.snapshot_degrees.append(torch.tensor(deg, dtype=torch.float))
                    self.snapshot_num_edges.append(adj.nnz // 2)
            except KeyError as e:
                print(f"Error: Missing key {e} in artifact file.")
                exit()

        self.min_history_len = min_history_len
        self.max_history_len = max_history_len

        if preloaded_p2_raw is not None:
            print(f"  - [Memory Opt] Using preloaded P2 data ({len(preloaded_p2_raw)} snapshots)...")
            self.embedding_sequence_csr = self._convert_raw_list_to_csr(preloaded_p2_raw, self.num_nodes_global)
        else:
            print(f"  - Loading P2 (Embed - Sparse Storage): {p2_pkl_path}")
            self.embedding_sequence_csr = self._load_embedding_sequence_as_sparse(
                p2_pkl_path, self.num_nodes_global
            )

        self.num_snapshots = len(self.embedding_sequence_csr)

        print(f"Creating training sample index for '{split}' split (Limit: {self.max_samples})...")
        self.samples = self._create_samples()
        print(f"Successfully created {len(self.samples)} subgraph samples for '{split}' split.")

        gc.collect()

    def _convert_raw_list_to_csr(self, raw_list, num_nodes_global):
        sparse_list = []
        print(f"  - Converting {len(raw_list)} snapshots to CSR format...")

        for item in tqdm(raw_list, desc="Sparse Conversion"):
            emb = item.get('embedding', None)
            if emb is None:
                continue
            if isinstance(emb, torch.Tensor):
                emb = emb.numpy()
            if sp.issparse(emb):
                sparse_list.append(emb.tocsr())
                continue
            indices = item.get('indices', None)
            if isinstance(indices, torch.Tensor):
                indices = indices.numpy()
            if indices is not None and emb.shape[0] == len(indices):
                dim = emb.shape[1]
                N = num_nodes_global
                rows = np.repeat(indices, dim)
                cols = np.tile(np.arange(dim), len(indices))
                data_flat = emb.flatten()
                mat = sp.csr_matrix((data_flat, (rows, cols)), shape=(N, dim), dtype=np.float32)
                sparse_list.append(mat)
            else:
                sparse_list.append(sp.csr_matrix(emb, dtype=np.float32))
        return sparse_list

    def _load_embedding_sequence_as_sparse(self, embedding_file: str, num_nodes_global: int):
        try:
            print("  - Reading P2 file...")
            try:
                data = joblib.load(embedding_file)
            except Exception:
                with open(embedding_file, 'rb') as f:
                    data = pickle.load(f)

            raw_list = data[self.split]
            del data
            gc.collect()

            return self._convert_raw_list_to_csr(raw_list, num_nodes_global)

        except Exception as e:
            print(f"Error processing embedding data structure: {e}")
            import traceback
            traceback.print_exc()
            exit()

    def _create_samples(self):
        candidates = []
        rng = np.random.default_rng(seed=42)
        print("  - Generating candidate sample pool...")
        for target_idx in range(self.min_history_len, self.num_snapshots):
            for horizon in self.forecast_horizons:
                adj_idx = target_idx - horizon
                if adj_idx < self.min_history_len - 1:
                    continue
                max_start_idx = max(0, adj_idx - self.min_history_len + 1)
                min_start_idx = max(0, adj_idx - self.max_history_len + 1)
                if min_start_idx > max_start_idx:
                    continue
                if min_start_idx == max_start_idx:
                    start_idx = min_start_idx
                else:
                    start_idx = rng.integers(min_start_idx, max_start_idx, endpoint=True)
                history_indices = list(range(start_idx, adj_idx + 1))
                current_num_nodes = self.snapshot_node_counts[adj_idx]
                if current_num_nodes <= self.subgraph_size_limit:
                    num_needed = 1
                else:
                    one_step_budget = int(np.ceil(
                        (current_num_nodes / self.subgraph_size_limit) * 1.2
                    ))
                    num_needed = max(
                        1, int(np.ceil(one_step_budget / len(self.forecast_horizons)))
                    )
                seeds = rng.choice(current_num_nodes, num_needed, replace=True)
                for seed_node in seeds:
                    candidates.append({
                        "history_indices": history_indices,
                        "target_idx": target_idx,
                        "seed_node": seed_node,
                        "adj_idx": adj_idx,
                        "forecast_horizon": horizon,
                    })
        total_candidates = len(candidates)
        print(f"  - Initial candidate sample count: {total_candidates}")
        if total_candidates > self.max_samples:
            print(f"  - Sample count exceeds limit ({self.max_samples}), performing horizon-balanced downsampling...")
            grouped = {horizon: [] for horizon in self.forecast_horizons}
            for candidate in candidates:
                grouped[candidate["forecast_horizon"]].append(candidate)
            for values in grouped.values():
                rng.shuffle(values)
            quota = self.max_samples // len(grouped)
            final_samples = []
            leftovers = []
            for horizon in self.forecast_horizons:
                values = grouped[horizon]
                final_samples.extend(values[:quota])
                leftovers.extend(values[quota:])
            rng.shuffle(leftovers)
            final_samples.extend(leftovers[: self.max_samples - len(final_samples)])
            rng.shuffle(final_samples)
        else:
            final_samples = candidates
        return final_samples

    def _calculate_dsm(self, num_nodes_sub, num_edges_sub, sum_degrees_sub, total_edges_graph):
        if num_nodes_sub == 0: return -float('inf')
        term1 = 1.0 / (2.0 * (num_nodes_sub ** self.dsm_tau))
        term2 = (2.0 * num_edges_sub) - (sum_degrees_sub ** 2) / (2.0 * total_edges_graph + 1e-9)
        return term1 * term2

    def _alice_subgraph_sampling(self, edge_index, start_node, max_limit, adj_idx):
        if not isinstance(start_node, torch.Tensor): seed = torch.tensor([start_node], dtype=torch.long)
        else: seed = start_node.view(-1).cpu()
        current_num_nodes = self.snapshot_node_counts[adj_idx]
        if seed.item() >= current_num_nodes: seed = torch.tensor([seed.item() % current_num_nodes], dtype=torch.long)
        if edge_index.numel() == 0: return seed
        all_degrees = self.snapshot_degrees[adj_idx]
        total_edges_graph = self.snapshot_num_edges[adj_idx]
        best_subset = seed
        best_dsm = -float('inf')
        max_hops = 2
        for k in range(1, max_hops + 1):
            subset, sub_edge_index, _, _ = k_hop_subgraph(
                node_idx=seed, num_hops=k, edge_index=edge_index, relabel_nodes=False, num_nodes=current_num_nodes
            )
            num_nodes_sub = subset.size(0)
            if num_nodes_sub > max_limit:
                if k == 1:
                    mask = subset != seed.item()
                    others = subset[mask]
                    perm = torch.randperm(others.size(0))
                    subset = torch.cat([seed, others[perm[:max_limit-1]]])
                    return subset
                else:
                    return best_subset
            num_edges_sub = sub_edge_index.size(1) // 2
            sum_degrees_sub = all_degrees[subset].sum().item()
            dsm_score = self._calculate_dsm(num_nodes_sub, num_edges_sub, sum_degrees_sub, total_edges_graph)
            if dsm_score > best_dsm:
                best_dsm = dsm_score
                best_subset = subset
        return best_subset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        history_indices = sample_info['history_indices']
        target_idx = sample_info['target_idx']
        raw_seed_node = sample_info['seed_node']
        adj_idx = sample_info['adj_idx']
        forecast_horizon = int(sample_info.get('forecast_horizon', target_idx - adj_idx))
        safe_adj_idx = max(0, adj_idx)
        adj_sparse = self.aligned_inputs[safe_adj_idx]['adj']
        current_graph_nodes = adj_sparse.shape[0]
        if raw_seed_node >= current_graph_nodes:
            seed_node = raw_seed_node % current_graph_nodes
        else:
            seed_node = raw_seed_node
        edge_index_full, _ = from_scipy_sparse_matrix(adj_sparse)
        subset_nodes = self._alice_subgraph_sampling(
            edge_index_full, seed_node, self.subgraph_size_limit, safe_adj_idx
        )
        edge_index_sub, _ = subgraph(
            subset_nodes, edge_index_full, relabel_nodes=True, num_nodes=current_graph_nodes
        )
        subset_nodes_np = subset_nodes.numpy()
        history_embs_list = []
        for i in history_indices:
            sub_emb = self.embedding_sequence_csr[i][subset_nodes_np].toarray()
            history_embs_list.append(torch.from_numpy(sub_emb))
        history_embeddings = torch.stack(history_embs_list)
        target_emb_np = self.embedding_sequence_csr[target_idx][subset_nodes_np].toarray()
        target_embedding = torch.from_numpy(target_emb_np)
        hist_ts_vals = [self.aligned_inputs[i]['timestamp'] for i in history_indices]
        history_timestamps = torch.tensor(hist_ts_vals, dtype=torch.long)
        last_history_timestamp = history_timestamps[-1]
        delta_t = torch.tensor(forecast_horizon, dtype=torch.long)
        history_relative_timestamps = history_timestamps - last_history_timestamp
        return {
            "history_embeddings": history_embeddings,
            "history_relative_timestamps": history_relative_timestamps,
            "target_embedding": target_embedding,
            "delta_t": delta_t,
            "edge_index": edge_index_sub,
            "node_indices": subset_nodes
        }
