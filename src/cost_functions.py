
from coarse_graph import CoarseGraph
import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple, Dict, Set, List

class ConvMatchCost:
    @staticmethod
    def _compute_gcn_conv(adj: sp.csr_matrix, features: np.ndarray, sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a_tilde = adj + sp.diags(sizes, format='csr')
        d_tilde_vec = np.array(a_tilde.sum(axis=1)).flatten()
        epsilon = 1e-8
        d_tilde_inv_sqrt_vec = 1.0 / np.sqrt(d_tilde_vec + epsilon)
        d_tilde_inv_sqrt = sp.diags(d_tilde_inv_sqrt_vec)
        norm_adj = d_tilde_inv_sqrt @ a_tilde @ d_tilde_inv_sqrt
        h_tilde = norm_adj @ features
        return h_tilde, d_tilde_vec

    def calculate(self, *args, **kwargs) -> float:
        return 0.0

class AConvMatchCost(ConvMatchCost):

    def calculate_batch_torch(self,
                              features: torch.Tensor,
                              d_tilde_vec: torch.Tensor,
                              sizes: torch.Tensor,
                              h_tilde: torch.Tensor,
                              pairs: torch.Tensor
                              ) -> torch.Tensor:
        epsilon = 1e-8

        u_idx = pairs[:, 0]
        v_idx = pairs[:, 1]

        x_u = features[u_idx]
        x_v = features[v_idx]

        c_u = sizes[u_idx].unsqueeze(1)
        c_v = sizes[v_idx].unsqueeze(1)

        d_u = d_tilde_vec[u_idx].unsqueeze(1)
        d_v = d_tilde_vec[v_idx].unsqueeze(1)

        c_uv = c_u + c_v

        numerator = x_u * c_u
        numerator.add_(x_v * c_v)
        x_uv = numerator / (c_uv + epsilon)

        d_uv = d_u + d_v

        h_u_before = h_tilde[u_idx]
        h_v_before = h_tilde[v_idx]

        h_uv_self_term = x_uv / (d_uv + epsilon)

        term_u = x_u / (d_u + epsilon)
        diff_u = h_uv_self_term - term_u

        cost_u = torch.sum(torch.abs(h_uv_self_term - term_u), dim=1)

        term_v = x_v / (d_v + epsilon)
        cost_v = torch.sum(torch.abs(h_uv_self_term - term_v), dim=1)

        return cost_u + cost_v

    def calculate_vectorized(self, adj: sp.csr_matrix, features: np.ndarray, sizes: np.ndarray,
                                s_u_id: int, s_v_id: int,
                                h_tilde_before: np.ndarray,
                                d_tilde_vec_before: np.ndarray) -> float:
        epsilon = 1e-8

        x_u, x_v = features[s_u_id], features[s_v_id]
        c_u, c_v = sizes[s_u_id], sizes[s_v_id]
        d_u, d_v = d_tilde_vec_before[s_u_id], d_tilde_vec_before[s_v_id]

        c_uv = c_u + c_v
        x_uv = (x_u * c_u + x_v * c_v) / (c_uv + epsilon)
        d_uv = d_u + d_v - c_u - c_v + c_uv

        cost = 0.0

        h_u_before = h_tilde_before[s_u_id]
        h_v_before = h_tilde_before[s_v_id]

        h_uv_self_term = x_uv / (d_uv + epsilon)

        term_u = x_u / (d_u + epsilon)
        cost += np.sum(np.abs(h_uv_self_term - term_u))

        term_v = x_v / (d_v + epsilon)
        cost += np.sum(np.abs(h_uv_self_term - term_v))

        u_start, u_end = adj.indptr[s_u_id], adj.indptr[s_u_id+1]
        v_start, v_end = adj.indptr[s_v_id], adj.indptr[s_v_id+1]

        u_indices = adj.indices[u_start:u_end]
        v_indices = adj.indices[v_start:v_end]

        all_neighbors = np.union1d(u_indices, v_indices)
        mask_self = (all_neighbors != s_u_id) & (all_neighbors != s_v_id)
        all_neighbors = all_neighbors[mask_self]

        if len(all_neighbors) == 0:
            return float(cost)

        D_I = d_tilde_vec_before[all_neighbors]
        X_I = features[all_neighbors]

        u_data = adj.data[u_start:u_end]
        v_data = adj.data[v_start:v_end]

        def get_weights(indices, data, targets):
            if len(indices) == 0: return np.zeros(len(targets))
            ptr = np.searchsorted(indices, targets)
            ptr = np.clip(ptr, 0, len(indices)-1)
            mask = (indices[ptr] == targets)
            res = np.zeros(len(targets))
            res[mask] = data[ptr[mask]]
            return res

        W_IU = get_weights(u_indices, u_data, all_neighbors)
        W_IV = get_weights(v_indices, v_data, all_neighbors)
        W_I_UV = W_IU + W_IV

        Sqrt_term_u = 1.0 / np.sqrt((D_I * d_u) + epsilon)
        Sqrt_term_v = 1.0 / np.sqrt((D_I * d_v) + epsilon)
        Sqrt_term_uv = 1.0 / np.sqrt((D_I * d_uv) + epsilon)

        msg_u = (W_IU * Sqrt_term_u)[:, None] * x_u[None, :]
        msg_v = (W_IV * Sqrt_term_v)[:, None] * x_v[None, :]
        msg_before = msg_u + msg_v

        msg_after = (W_I_UV * Sqrt_term_uv)[:, None] * x_uv[None, :]

        cost += np.sum(np.abs(msg_after - msg_before))

        return float(cost)

    def calculate(self, graph_snapshot: CoarseGraph, s_u_id: int, s_v_id: int,
                  h_tilde_before: np.ndarray,
                  d_tilde_vec_before: np.ndarray = None,
                  neighbor_weights_map: Dict[int, Dict[int, float]] = None) -> float:
        if sp.isspmatrix_csr(graph_snapshot.adj) and d_tilde_vec_before is not None:
            return self.calculate_vectorized(
                graph_snapshot.adj, graph_snapshot.features, graph_snapshot.supernode_sizes,
                s_u_id, s_v_id, h_tilde_before, d_tilde_vec_before
            )
        return 0.0
