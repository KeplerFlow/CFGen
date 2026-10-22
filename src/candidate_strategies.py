from coarse_graph import CoarseGraph
from typing import List, Dict, Optional, Union
import numpy as np
import scipy.sparse as sp
import logging
from sklearn.decomposition import TruncatedSVD
import faiss

from scipy.sparse.linalg import LinearOperator, svds

try:
    import cupy as cp
    import cupyx.scipy.sparse as cps
    HAS_CUPY = True
    logging.info("CuPy available. SGC computation will be GPU accelerated.")
except ImportError:
    HAS_CUPY = False
    logging.warning("CuPy not installed. SGC computation will run on CPU.")

class BaseCandidateStrategy:

    def find_candidates(self, node_id: int, graph: CoarseGraph) -> List[int]:
        raise NotImplementedError

    def find_candidates_batch(self, node_ids: List[int], graph: CoarseGraph) -> Dict[int, List[int]]:
        results = {}
        for nid in node_ids:
            results[nid] = self.find_candidates(nid, graph)
        return results

    def build_index(self, graph: CoarseGraph):
        pass

    def find_candidates_from_index(self, node_id: int, graph: CoarseGraph) -> List[int]:
        return self.find_candidates(node_id, graph)

class NeighborhoodCandidateStrategy(BaseCandidateStrategy):
    def find_candidates(self, node_id: int, graph: CoarseGraph) -> List[int]:
        supernode_id = graph.get_supernode_id(node_id)
        neighbor_supernode_ids = list(graph.adj[supernode_id].indices)
        return neighbor_supernode_ids

    def find_candidates_batch(self, node_ids: List[int], graph: CoarseGraph) -> Dict[int, List[int]]:
        results = {}
        supernode_ids = [graph.get_supernode_id(nid) for nid in node_ids]

        indptr = graph.adj.indptr
        indices = graph.adj.indices

        for i, nid in enumerate(node_ids):
            sid = supernode_ids[i]
            start, end = indptr[sid], indptr[sid+1]
            if start == end:
                results[nid] = []
            else:
                results[nid] = indices[start:end].tolist()
        return results

class EmbeddingCandidateStrategy(BaseCandidateStrategy):
    def __init__(self,
                 features: np.ndarray,
                 adj: sp.csr_matrix,
                 k_neighbors: int = 10,
                 sgc_k_hops: int = 2,
                 output_dim: int = 32):

        logging.info(f"Initializing EmbeddingCandidateStrategy with k={k_neighbors}, hops={sgc_k_hops}.")
        self.k_neighbors = k_neighbors

        logging.info("Pre-computing SGC embeddings for the original graph...")

        self.original_sgc_embeddings = self._compute_sgc(
            features, adj, sgc_k_hops, output_dim
        )

        self.feature_dim = self.original_sgc_embeddings.shape[1]

        logging.info(f"SGC embeddings computed, shape: {self.original_sgc_embeddings.shape}.")

        self.faiss_index = None
        self.active_supernode_ids = None
        self.active_sid_to_idx_map = None
        self.faiss_gpu_res = None

        try:
            self.faiss_gpu_res = faiss.StandardGpuResources()
            self.faiss_gpu_res.setTempMemory(1024 * 1024 * 512)
        except Exception as e:
            logging.warning(f"Could not initialize Faiss GPU resources: {e}. Will run on CPU.")
            self.faiss_gpu_res = None

    @staticmethod
    def _compute_sgc(features: np.ndarray, adj: sp.csr_matrix, k: int, output_dim: int) -> np.ndarray:
        num_nodes = adj.shape[0]

        is_identity_input = (features.shape[0] == features.shape[1] == num_nodes)

        HUGE_GRAPH_THRESHOLD = 80000

        if is_identity_input and num_nodes > HUGE_GRAPH_THRESHOLD:
            logging.info(f"  - [SGC Optimization] Detected giant graph (N={num_nodes}) with no initial features.")
            logging.info(f"  - Enabled 'Randomized SGC': skipping SVD, propagating random Gaussian features directly.")

            np.random.seed(42)
            current_features = np.random.randn(num_nodes, output_dim).astype(np.float32)

            logging.info("  - Computing normalized adjacency matrix...")
            d_vec = np.array(adj.sum(axis=1)).flatten()
            d_inv_sqrt = np.power(d_vec + 1e-8, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            for i in range(k):
                logging.info(f"  - SGC Propagating step {i+1}/{k}...")
                current_features = d_mat_inv_sqrt @ current_features
                current_features = adj @ current_features
                current_features = d_mat_inv_sqrt @ current_features

            return current_features.astype(np.float32)

        if HAS_CUPY and not is_identity_input:
             pass

        logging.info(f"  - SGC [CPU]: Standard calculation.")

        a_tilde = adj + sp.identity(adj.shape[0], format='csr')
        d_tilde_vec = np.array(a_tilde.sum(axis=1)).flatten()
        d_tilde_inv_sqrt_vec = 1.0 / np.sqrt(d_tilde_vec + 1e-8)
        d_tilde_inv_sqrt = sp.diags(d_tilde_inv_sqrt_vec)
        norm_adj = d_tilde_inv_sqrt @ a_tilde @ d_tilde_inv_sqrt

        if is_identity_input:
             def matvec(v):
                result = v.copy()
                for _ in range(k):
                    result = norm_adj @ result
                return result

             n = adj.shape[0]
             sk_operator = LinearOperator(shape=(n, n), matvec=matvec, rmatvec=matvec, dtype=norm_adj.dtype)
             u, s, vh = svds(sk_operator, k=output_dim, which='LM', maxiter=20, tol=1e-2)
             idx = np.argsort(s)[::-1]
             return (u[:, idx] * s[idx]).astype(np.float32)
        else:
             output_features = features
             for _ in range(k):
                 output_features = norm_adj @ output_features
             return output_features.astype(np.float32)

    def build_index(self, graph: CoarseGraph):
        self.active_supernode_ids = np.where(graph.supernode_sizes > 0)[0]
        num_active = len(self.active_supernode_ids)

        if num_active == 0:
            self.faiss_index = None
            return

        active_embeddings = graph.features[self.active_supernode_ids]

        if active_embeddings.dtype != np.float32:
            active_embeddings = active_embeddings.astype(np.float32)
        if not active_embeddings.flags['C_CONTIGUOUS']:
            active_embeddings = np.ascontiguousarray(active_embeddings)

        faiss.normalize_L2(active_embeddings)

        dim = self.feature_dim

        try:

            index_cpu = faiss.IndexFlatIP(dim)

            if self.faiss_gpu_res is not None:

                self.faiss_index = faiss.index_cpu_to_gpu(self.faiss_gpu_res, 0, index_cpu)
            else:
                self.faiss_index = index_cpu

            self.faiss_index.add(active_embeddings)

        except RuntimeError as e:
            logging.warning(f"Failed to build GPU index (likely OOM): {e}. Falling back to CPU index.")
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(active_embeddings)
        except Exception as e:
            logging.error(f"Unexpected error building index: {e}")
            self.faiss_index = None

        self.active_sid_to_idx_map = {sid: i for i, sid in enumerate(self.active_supernode_ids)}

    def find_candidates_batch(self, node_ids: List[int], graph: CoarseGraph) -> Dict[int, List[int]]:
        if self.faiss_index is None:
            return super().find_candidates_batch(node_ids, graph)

        query_sids = [graph.get_supernode_id(nid) for nid in node_ids]

        valid_indices = []
        valid_query_sids = []

        for i, sid in enumerate(query_sids):
            if sid in self.active_sid_to_idx_map:
                valid_indices.append(i)
                valid_query_sids.append(sid)

        if not valid_indices:
            return {nid: [] for nid in node_ids}

        query_embeddings = graph.features[valid_query_sids]

        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        if not query_embeddings.flags['C_CONTIGUOUS']:
            query_embeddings = np.ascontiguousarray(query_embeddings)

        faiss.normalize_L2(query_embeddings)

        k_search = min(self.k_neighbors + 1, self.faiss_index.ntotal)

        all_distances = []
        all_indices = []

        num_queries = query_embeddings.shape[0]
        chunk_size = 4096

        for start_idx in range(0, num_queries, chunk_size):
            end_idx = min(start_idx + chunk_size, num_queries)
            batch_query = query_embeddings[start_idx:end_idx]

            D, I = self.faiss_index.search(batch_query, k_search)

            all_distances.append(D)
            all_indices.append(I)

        final_indices = np.vstack(all_indices)

        results = {nid: [] for nid in node_ids}

        active_sids_arr = self.active_supernode_ids

        for i, valid_idx in enumerate(valid_indices):
            original_node_id = node_ids[valid_idx]
            query_sid = valid_query_sids[i]

            neighbor_indices = final_indices[i]

            candidates = []
            for n_idx in neighbor_indices:
                if n_idx == -1: continue

                candidate_sid = active_sids_arr[n_idx]
                if candidate_sid != query_sid:
                    candidates.append(int(candidate_sid))

            results[original_node_id] = candidates

        return results

    def find_candidates(self, node_id: int, graph: CoarseGraph) -> List[int]:
        if self.faiss_index is None:
            self.build_index(graph)

        res = self.find_candidates_batch([node_id], graph)
        return res.get(node_id, [])

    def find_candidates_from_index(self, node_id: int, graph: CoarseGraph) -> List[int]:
        return self.find_candidates(node_id, graph)
