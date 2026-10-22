import numpy as np
import scipy.sparse as sp
import logging
from typing import Union, Tuple, List, Set
import copy

class CoarseGraph:
    def __init__(self, initial_adj: sp.csr_matrix, initial_features: np.ndarray):
        if not sp.isspmatrix_csr(initial_adj):
            raise TypeError("Initial adjacency matrix must be in CSR format.")

        self.num_original_nodes = initial_adj.shape[0]

        if initial_adj.shape[0] != initial_features.shape[0]:
            raise ValueError(f"Adj shape {initial_adj.shape} and Feature shape {initial_features.shape} mismatch on dim 0.")

        self.current_base_adj = initial_adj.copy().tocsr()
        self.original_features = initial_features.copy()

        self.adj = initial_adj.copy().asfptype().tocsr()
        self.degree = np.array(self.adj.sum(axis=1)).flatten()
        self.features = initial_features.copy()
        self.node_to_supernode = np.arange(self.num_original_nodes, dtype=int)
        self.supernode_to_nodes = {i: [i] for i in range(self.num_original_nodes)}
        self.supernode_sizes = np.ones(self.num_original_nodes, dtype=int)

        logging.info(f"CoarseGraph initialized with num_original_nodes={self.num_original_nodes}.")

    def update_base_adj(self, added_edges, deleted_edges):
        if not added_edges and not deleted_edges:
            return

        self.current_base_adj = self.current_base_adj.tolil()
        for u, v in added_edges:
            self.current_base_adj[u, v] = 1
            self.current_base_adj[v, u] = 1
        for u, v in deleted_edges:
            self.current_base_adj[u, v] = 0
            self.current_base_adj[v, u] = 0
        self.current_base_adj = self.current_base_adj.tocsr()
        self.current_base_adj.eliminate_zeros()

    def get_supernode_id(self, node_id: int) -> int:
        return self.node_to_supernode[node_id]

    def _merge_supernodes_internal_LIL(self, s_u_id: int, s_v_id: int):
        self.adj[s_u_id, :] += self.adj[s_v_id, :]

        self.adj[s_v_id, :] = 0
        self.adj[:, s_v_id] = 0

        self.adj[:, s_u_id] = self.adj[s_u_id, :].T

    def merge_supernodes(self, s_u_id: int, s_v_id: int) -> int:
        if s_u_id == s_v_id: return s_u_id

        if self.supernode_sizes[s_v_id] == 0:
            return s_u_id
        if self.supernode_sizes[s_u_id] == 0:
            logging.warning(f"Warning: Supernode {s_u_id} (target) is empty. This is unusual.")

        nodes_to_move = self.supernode_to_nodes.pop(s_v_id, [])
        for node in nodes_to_move:
            self.node_to_supernode[node] = s_u_id

        if s_u_id not in self.supernode_to_nodes:
             self.supernode_to_nodes[s_u_id] = []
        self.supernode_to_nodes[s_u_id].extend(nodes_to_move)

        old_size_u = self.supernode_sizes[s_u_id]
        old_size_v = self.supernode_sizes[s_v_id]
        new_size = old_size_u + old_size_v

        if new_size > 0:
            self.features[s_u_id, :] = (self.features[s_u_id, :] * old_size_u + self.features[s_v_id, :] * old_size_v) / new_size
        else:
            self.features[s_u_id, :] = 0

        self.features[s_v_id, :] = 0
        self.supernode_sizes[s_u_id] = new_size
        self.supernode_sizes[s_v_id] = 0

        self.adj = self.adj.tolil()

        self._merge_supernodes_internal_LIL(s_u_id, s_v_id)

        self.adj = self.adj.tocsr()

        self.adj.setdiag(self.adj.diagonal())

        self.degree[s_u_id] = self.adj[s_u_id].sum()
        self.degree[s_v_id] = 0

        return s_u_id

    def merge_supernodes(self, s_u_id: int, s_v_id: int) -> int:
        if s_u_id == s_v_id: return s_u_id

        if self.supernode_sizes[s_v_id] == 0:
            return s_u_id
        if self.supernode_sizes[s_u_id] == 0:
            logging.warning(f"Warning: Supernode {s_u_id} (target) is empty. This is unusual.")

        nodes_to_move = self.supernode_to_nodes.pop(s_v_id, [])
        for node in nodes_to_move:
            self.node_to_supernode[node] = s_u_id

        if s_u_id not in self.supernode_to_nodes:
             self.supernode_to_nodes[s_u_id] = []
        self.supernode_to_nodes[s_u_id].extend(nodes_to_move)

        old_size_u = self.supernode_sizes[s_u_id]
        old_size_v = self.supernode_sizes[s_v_id]
        new_size = old_size_u + old_size_v

        if new_size > 0:
            self.features[s_u_id, :] = (self.features[s_u_id, :] * old_size_u + self.features[s_v_id, :] * old_size_v) / new_size
        else:
            self.features[s_u_id, :] = 0

        self.features[s_v_id, :] = 0
        self.supernode_sizes[s_u_id] = new_size
        self.supernode_sizes[s_v_id] = 0

        self.adj = self.adj.tolil()
        self._merge_supernodes_internal_LIL(s_u_id, s_v_id)
        self.adj = self.adj.tocsr()

        if self.adj[s_u_id, s_u_id] == 0 and new_size > 0:
             self.adj[s_u_id, s_u_id] = 1.0

        self.degree[s_u_id] = self.adj[s_u_id].sum()
        self.degree[s_v_id] = 0

        return s_u_id

    def split_supernode(self, node_id_to_split: int) -> Union[Tuple[int, int], None]:
        current_sid = self.get_supernode_id(node_id_to_split)

        if self.supernode_sizes[current_sid] <= 1:
            return None

        new_sid = self._get_available_supernode_id()

        self.supernode_to_nodes[current_sid].remove(node_id_to_split)
        self.supernode_to_nodes[new_sid] = [node_id_to_split]
        self.node_to_supernode[node_id_to_split] = new_sid
        self.supernode_sizes[current_sid] -= 1
        self.supernode_sizes[new_sid] = 1

        remaining_nodes = self.supernode_to_nodes[current_sid]
        if remaining_nodes:
             self.features[current_sid, :] = self.original_features[remaining_nodes].mean(axis=0)
        else:
             self.features[current_sid, :] = 0
        self.features[new_sid, :] = self.original_features[node_id_to_split]

        self.adj = self.adj.tolil()

        nodes_in_new_sid = self.supernode_to_nodes[new_sid]
        nodes_in_current_sid = self.supernode_to_nodes[current_sid]

        mapping = self.node_to_supernode

        base_row_new = sp.csr_matrix(
            self.current_base_adj[nodes_in_new_sid, :].sum(axis=0)
        ).tocoo()

        connected_supernodes_new = mapping[base_row_new.col]

        new_coarse_row = np.bincount(
            connected_supernodes_new,
            weights=base_row_new.data,
            minlength=self.num_original_nodes
        )

        if nodes_in_current_sid:
            base_row_current = sp.csr_matrix(
                self.current_base_adj[nodes_in_current_sid, :].sum(axis=0)
            ).tocoo()

            connected_supernodes_curr = mapping[base_row_current.col]
            current_coarse_row = np.bincount(
                connected_supernodes_curr,
                weights=base_row_current.data,
                minlength=self.num_original_nodes
            )
        else:
            current_coarse_row = np.zeros(self.num_original_nodes)

        self.adj[new_sid, :] = new_coarse_row
        self.adj[:, new_sid] = new_coarse_row.T

        self.adj[current_sid, :] = current_coarse_row
        self.adj[:, current_sid] = current_coarse_row.T

        self.adj = self.adj.tocsr()

        self.adj.setdiag(self.adj.diagonal())

        self.degree[current_sid] = self.adj[current_sid].sum()
        self.degree[new_sid] = self.adj[new_sid].sum()

        return current_sid, new_sid

    def split_supernode_LIL(self, node_id_to_split: int) -> Union[Tuple[int, int], None]:
        current_sid = self.get_supernode_id(node_id_to_split)

        if self.supernode_sizes[current_sid] <= 1:
            return None

        new_sid = self._get_available_supernode_id()

        self.supernode_to_nodes[current_sid].remove(node_id_to_split)
        self.supernode_to_nodes[new_sid] = [node_id_to_split]
        self.node_to_supernode[node_id_to_split] = new_sid
        self.supernode_sizes[current_sid] -= 1
        self.supernode_sizes[new_sid] = 1

        remaining_nodes = self.supernode_to_nodes[current_sid]
        if remaining_nodes:
             self.features[current_sid, :] = self.original_features[remaining_nodes].mean(axis=0)
        else:
             self.features[current_sid, :] = 0
        self.features[new_sid, :] = self.original_features[node_id_to_split]

        nodes_in_new_sid = self.supernode_to_nodes[new_sid]
        nodes_in_current_sid = self.supernode_to_nodes[current_sid]
        mapping = self.node_to_supernode

        base_row_new = sp.csr_matrix(
            self.current_base_adj[nodes_in_new_sid, :].sum(axis=0)
        ).tocoo()
        connected_supernodes_new = mapping[base_row_new.col]
        new_coarse_row = np.bincount(
            connected_supernodes_new,
            weights=base_row_new.data,
            minlength=self.num_original_nodes
        )

        if nodes_in_current_sid:
            base_row_current = sp.csr_matrix(
                self.current_base_adj[nodes_in_current_sid, :].sum(axis=0)
            ).tocoo()
            connected_supernodes_curr = mapping[base_row_current.col]
            current_coarse_row = np.bincount(
                connected_supernodes_curr,
                weights=base_row_current.data,
                minlength=self.num_original_nodes
            )
        else:
            current_coarse_row = np.zeros(self.num_original_nodes)

        self.adj[new_sid, :] = new_coarse_row
        self.adj[:, new_sid] = new_coarse_row.T
        self.adj[current_sid, :] = current_coarse_row
        self.adj[:, current_sid] = current_coarse_row.T

        self.adj.setdiag(self.adj.diagonal())

        self.degree[current_sid] = current_coarse_row.sum()
        self.degree[new_sid] = new_coarse_row.sum()

        return current_sid, new_sid

    def split_supernodes_batch_CSR(self, node_ids_to_split: List[int]) -> Set[int]:
        available_sids = np.where(self.supernode_sizes == 0)[0]

        valid_nodes_to_split = []
        affected_sids = set()
        for node_id in node_ids_to_split:
            current_sid = self.get_supernode_id(node_id)
            affected_sids.add(current_sid)
            if self.supernode_sizes[current_sid] > 1:
                valid_nodes_to_split.append(node_id)

        if not valid_nodes_to_split:
            logging.info("  - (Batch Split) 1/3: No nodes need to be actually split.")
            return affected_sids

        if len(available_sids) < len(valid_nodes_to_split):
            logging.warning(f"  - (Batch Split) Warning: Insufficient empty supernodes available. Skipping split.")
            return affected_sids

        sid_pool = iter(available_sids)

        logging.info(f"  - (Batch Split) 1/3: Updating mapping for {len(valid_nodes_to_split)} nodes...")

        for node_id in valid_nodes_to_split:
            current_sid = self.get_supernode_id(node_id)
            new_sid = next(sid_pool)

            self.supernode_to_nodes[current_sid].remove(node_id)
            self.supernode_to_nodes[new_sid] = [node_id]
            self.node_to_supernode[node_id] = new_sid

            affected_sids.add(new_sid)

        logging.info(f"  - (Batch Split) 2/3: Updating features for {len(affected_sids)} affected supernodes...")

        for sid in affected_sids:
            node_list = self.supernode_to_nodes.get(sid, [])
            if node_list:
                self.features[sid, :] = self.original_features[node_list].mean(axis=0)
                self.supernode_sizes[sid] = len(node_list)
            else:
                self.features[sid, :] = 0
                self.supernode_sizes[sid] = 0

        logging.info(f"  - (Batch Split) 3/3: Batch rebuilding the entire coarse adjacency matrix (1 pass)...")

        base_adj_coo = self.current_base_adj.tocoo()
        s_row = self.node_to_supernode[base_adj_coo.row]
        s_col = self.node_to_supernode[base_adj_coo.col]

        new_adj = sp.csr_matrix(
            (base_adj_coo.data, (s_row, s_col)),
            shape=(self.num_original_nodes, self.num_original_nodes)
        )

        diag = new_adj.diagonal()
        active_indices = np.where(self.supernode_sizes > 0)[0]

        new_diag = diag.copy()
        new_diag[active_indices] = np.maximum(new_diag[active_indices], 1.0)

        new_adj.setdiag(new_diag)

        self.adj = new_adj
        self.degree = np.array(new_adj.sum(axis=1)).flatten()

        logging.info(f"  - (Batch Split) 3/3: Batch rebuild complete.")

        return affected_sids
    def _get_available_supernode_id(self) -> int:
        available_ids = np.where(self.supernode_sizes == 0)[0]
        if len(available_ids) > 0:
            return available_ids[0]
        else:
            raise RuntimeError("No available supernode IDs to use for splitting.")
