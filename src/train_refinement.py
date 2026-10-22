
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import gc
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import joblib
import argparse

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import GATv2Conv
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SimilarityConsistencyLoss(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp

    def forward(self, pred_emb, gt_emb):
        pred_norm = F.normalize(pred_emb, p=2, dim=1)
        gt_norm = F.normalize(gt_emb, p=2, dim=1)

        sim_pred = torch.mm(pred_norm, pred_norm.t()) / self.temp
        sim_gt = torch.mm(gt_norm, gt_norm.t()) / self.temp

        loss = F.mse_loss(sim_pred, sim_gt)

        return loss

class CommunityContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp

    def forward(self, z, comm):
        if comm is None or comm.numel() < 2:
            return z.new_zeros(())
        z_norm = F.normalize(z, dim=1)
        sim = z_norm @ z_norm.t() / self.temp
        same = comm.unsqueeze(0) == comm.unsqueeze(1)
        same.fill_diagonal_(False)
        valid = torch.ones_like(same, dtype=torch.bool); valid.fill_diagonal_(False)
        exp = torch.exp(sim) * valid
        log_prob = sim - torch.log(exp.sum(1) + 1e-9)
        has_pos = same.any(1)
        if not has_pos.any():
            return z.new_zeros(())
        pos = (log_prob * same.float()).sum(1) / (same.float().sum(1) + 1e-9)
        return -(pos[has_pos].mean())

class EmbeddingRefinementGNN_V15_1(nn.Module):
    def __init__(self, embed_dim: int, n_layers: int = 3, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        self.layers.append(GATv2Conv(embed_dim, embed_dim, heads=heads, concat=True))
        self.norms.append(nn.LayerNorm(embed_dim * heads))

        for _ in range(n_layers - 2):
            self.layers.append(GATv2Conv(embed_dim * heads, embed_dim, heads=heads, concat=True))
            self.norms.append(nn.LayerNorm(embed_dim * heads))

        self.layers.append(GATv2Conv(embed_dim * heads, embed_dim, heads=1, concat=False))

    def forward(self, x_coarse, edge_index):
        identity = x_coarse
        h = x_coarse
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = self.norms[i](h)
                h = F.elu(h)
        return identity + h

class RefinementDataset_V15(Dataset):
    def __init__(self,
                 phase1_pkl_path: str,
                 p2_coarse_pkl_path: str,
                 p2_golden_pkl_path: str,
                 split: str):

        if split == 'train':
            print(f"[Info] Loading Refinement Dataset (Train/Val)...")

        try:
            phase1_data = joblib.load(phase1_pkl_path)
        except Exception:
            with open(phase1_pkl_path, 'rb') as f:
                phase1_data = pickle.load(f)

        self.raw_adj_list = [item['adj'] for item in phase1_data[f'{split}_aligned_inputs']]
        raw_partition_history = phase1_data[f'{split}_partition_history']
        self.num_total_nodes = phase1_data['num_total_nodes']

        del phase1_data
        gc.collect()

        try:
            p2_coarse_data = joblib.load(p2_coarse_pkl_path)
        except Exception:
            with open(p2_coarse_pkl_path, 'rb') as f:
                p2_coarse_data = pickle.load(f)

        split_coarse_data = p2_coarse_data[split]

        self.compact_coarse_embeddings = []
        self.compact_edge_indices = []
        self.node_mappings = []

        for idx, item in enumerate(tqdm(split_coarse_data, desc="Compacting", disable=True)):
            vals = item.get('embedding')
            indices = item.get('indices')

            if vals is None: raise ValueError(f"Snapshot {idx} Coarse data corrupted")

            if isinstance(vals, np.ndarray): vals = torch.from_numpy(vals).float()
            if isinstance(indices, np.ndarray): indices = torch.from_numpy(indices).long()

            raw_adj = self.raw_adj_list[idx]

            if indices is not None:
                sorted_indices_np = indices.numpy()
                if not np.all(sorted_indices_np[:-1] <= sorted_indices_np[1:]):
                    sorted_indices_np = np.sort(sorted_indices_np)

                self.node_mappings.append(sorted_indices_np)
                self.compact_coarse_embeddings.append(vals)

                coo = raw_adj.tocoo()
                rows, cols = coo.row, coo.col
                mask = np.isin(rows, sorted_indices_np) & np.isin(cols, sorted_indices_np)
                valid_rows = rows[mask]
                valid_cols = cols[mask]

                remapped_rows = np.searchsorted(sorted_indices_np, valid_rows)
                remapped_cols = np.searchsorted(sorted_indices_np, valid_cols)

                edge_index = torch.stack([
                    torch.from_numpy(remapped_rows).long(),
                    torch.from_numpy(remapped_cols).long()
                ], dim=0)
                self.compact_edge_indices.append(edge_index)
            else:
                self.node_mappings.append(None)
                self.compact_coarse_embeddings.append(vals)
                edge_index, _ = from_scipy_sparse_matrix(raw_adj)
                self.compact_edge_indices.append(edge_index)

        del p2_coarse_data
        del split_coarse_data
        gc.collect()

        self.golden_supernode_labels = []

        try:
            p2_golden_data = joblib.load(p2_golden_pkl_path)
        except Exception:
            with open(p2_golden_pkl_path, 'rb') as f:
                p2_golden_data = pickle.load(f)

        snapshot_list = p2_golden_data[split]

        for idx in tqdm(range(len(snapshot_list)), desc="Labels", disable=True):
            golden_item = snapshot_list[idx]
            golden_vals = golden_item.get('embedding')
            golden_indices = golden_item.get('indices')

            partition_map = raw_partition_history[idx]
            active_supernodes = self.node_mappings[idx]
            num_compact_nodes = self.compact_coarse_embeddings[idx].shape[0]

            if golden_vals is None:
                self.golden_supernode_labels.append(
                    self.compact_coarse_embeddings[idx].detach().clone().float()
                )
                continue

            if isinstance(golden_vals, np.ndarray): golden_vals = torch.from_numpy(golden_vals).float()

            if active_supernodes is not None:
                max_global_id = int(active_supernodes.max()) if len(active_supernodes) > 0 else 0
                lookup_table = np.full(max_global_id + 1, -1, dtype=np.int32)
                compact_ids = np.arange(len(active_supernodes), dtype=np.int32)
                lookup_table[active_supernodes] = compact_ids

                if golden_indices is not None:
                    if isinstance(golden_indices, torch.Tensor): golden_indices = golden_indices.numpy()
                    relevant_global_sids = partition_map[golden_indices]
                    relevant_golden_emb = golden_vals
                else:
                    relevant_global_sids = partition_map
                    relevant_golden_emb = golden_vals

                target_compact_ids = np.full(len(relevant_global_sids), -1, dtype=np.int32)
                mask_in_range = relevant_global_sids <= max_global_id
                target_compact_ids[mask_in_range] = lookup_table[relevant_global_sids[mask_in_range]]

                p_tensor = torch.from_numpy(target_compact_ids).long()
                g_tensor = relevant_golden_emb

                mask = p_tensor >= 0
                p_tensor = p_tensor[mask]
                g_tensor = g_tensor[mask]
                del lookup_table

            else:
                if golden_indices is not None:
                    if isinstance(golden_indices, torch.Tensor): golden_indices = golden_indices.numpy()
                    p_tensor = torch.from_numpy(partition_map[golden_indices]).long()
                    g_tensor = golden_vals
                else:
                    p_tensor = torch.from_numpy(partition_map).long()
                    g_tensor = golden_vals

            if num_compact_nodes > 0 and g_tensor.shape[0] > 0:
                supernode_sum = torch.zeros(num_compact_nodes, g_tensor.shape[1], dtype=torch.float32)
                p_tensor = p_tensor.clamp(0, num_compact_nodes - 1)
                supernode_sum.index_add_(0, p_tensor, g_tensor)
                counts = torch.bincount(p_tensor, minlength=num_compact_nodes).float().clamp(min=1).unsqueeze(1)
                label = supernode_sum / counts
            else:
                label = torch.zeros(num_compact_nodes, g_tensor.shape[1] if g_tensor.ndim > 1 else 32)

            self.golden_supernode_labels.append(label)

        del p2_golden_data
        del raw_partition_history
        gc.collect()

        self.num_snapshots = len(self.compact_coarse_embeddings)
        import networkx as _nx
        try:
            from networkx.algorithms.community import louvain_communities as _louvain
        except Exception:
            _louvain = None
        self.comm_labels = []
        for _idx in range(len(self.compact_edge_indices)):
            _ei = self.compact_edge_indices[_idx]
            _nc = self.compact_coarse_embeddings[_idx].shape[0]
            if _louvain is not None and _ei.shape[1] > 0:
                _G = _nx.Graph(); _G.add_nodes_from(range(_nc))
                _G.add_edges_from(_ei.t().cpu().tolist())
                _comms = _louvain(_G, seed=0)
                _n2c = {int(_n): _i for _i, _c in enumerate(_comms) for _n in _c}
                _y_comm = torch.tensor([_n2c.get(_n, 0) for _n in range(_nc)], dtype=torch.long)
            else:
                _y_comm = torch.zeros(_nc, dtype=torch.long)
            self.comm_labels.append(_y_comm)
        self.degrees = []
        for _i, _ei in enumerate(self.compact_edge_indices):
            _nc = self.compact_coarse_embeddings[_i].shape[0]
            _d = torch.zeros(_nc, dtype=torch.float)
            if _ei.shape[1] > 0:
                _src = _ei[0].cpu().long()
                _d.scatter_add_(0, _src, torch.ones(_src.shape[0]))
            self.degrees.append(_d)

    def __len__(self):
        return self.num_snapshots

    def __getitem__(self, idx):
        return Data(
            x=self.compact_coarse_embeddings[idx],
            edge_index=self.compact_edge_indices[idx],
            y=self.golden_supernode_labels[idx],
            y_comm=self.comm_labels[idx],
            deg=self.degrees[idx]
        )

def run_one_epoch_subgraph(model, dataset, optimizer,
                           mse_loss_fn, sim_loss_fn, comm_loss_fn,
                           device, is_training=True, batch_size=1024):
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    SIM_LOSS_WEIGHT = 0.5
    NOISE_STD = 0.15 if is_training else 0.0

    context = torch.enable_grad() if is_training else torch.no_grad()

    snapshot_pbar = tqdm(range(len(dataset)), desc="Snapshots", leave=False, disable=True)

    for snapshot_idx in snapshot_pbar:
        full_snapshot_data = dataset[snapshot_idx]

        if full_snapshot_data.x.shape[0] == 0:
            continue

        loader = NeighborLoader(
            full_snapshot_data,
            num_neighbors=[10] * 3,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=0
        )

        batch_pbar = tqdm(loader, desc=f"Snap {snapshot_idx} Subgraphs", leave=False, disable=True)

        with context:
            for subgraph_batch in batch_pbar:
                subgraph_batch = subgraph_batch.to(device)

                num_target_nodes = subgraph_batch.batch_size

                if NOISE_STD > 0:
                    noise = torch.randn_like(subgraph_batch.x) * NOISE_STD
                    noisy_input = subgraph_batch.x + noise
                else:
                    noisy_input = subgraph_batch.x

                if is_training:
                    optimizer.zero_grad()

                out = model(noisy_input, subgraph_batch.edge_index)

                pred = out[:num_target_nodes]
                target = subgraph_batch.y[:num_target_nodes]

                loss_mse = mse_loss_fn(pred, target)
                loss_sim = sim_loss_fn(pred, target)
                _comm = subgraph_batch.y_comm[:num_target_nodes] if hasattr(subgraph_batch, 'y_comm') else None
                loss_comm = comm_loss_fn(pred, _comm) if comm_loss_fn is not None else pred.new_zeros(())
                _deg = subgraph_batch.deg[:num_target_nodes].float() if hasattr(subgraph_batch, 'deg') else None
                if _deg is not None and pred.shape[0] > 1 and _deg.max() > 0:
                    _pn = F.normalize(pred, dim=1)
                    _psim = _pn @ _pn.t()
                    _dn = _deg / (_deg.max() + 1e-9)
                    _dsim = _dn.unsqueeze(1) * _dn.unsqueeze(0)
                    loss_deg = F.mse_loss(_psim, _dsim)
                else:
                    loss_deg = pred.new_zeros(())
                loss = loss_mse + SIM_LOSS_WEIGHT * loss_sim + 0.5 * loss_comm + 0.5 * loss_deg

                if is_training:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                total_batches += 1

    avg_loss = 0.0 if total_batches == 0 else total_loss / total_batches

    return avg_loss

class AggressiveEarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, save_path='checkpoint.pt', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

def train_refinement_model_v15(config):
    train_dataset = RefinementDataset_V15(
        config['p1_path'], config['p2_coarse_path'], config['p2_golden_path'], split='train'
    )
    val_dataset = RefinementDataset_V15(
        config['p1_path'], config['p2_coarse_path'], config['p2_golden_path'], split='val'
    )

    if len(train_dataset) > 0:
        sample_data = train_dataset[0]
        embed_dim = sample_data.x.shape[1]
    else:
        embed_dim = 64

    model = EmbeddingRefinementGNN_V15_1(embed_dim=embed_dim).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    mse_loss_fn = nn.MSELoss().to(config['device'])
    sim_loss_fn = SimilarityConsistencyLoss(temp=1.0).to(config['device'])
    comm_loss_fn = CommunityContrastiveLoss(temp=0.5).to(config['device'])

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=False)

    early_stopping = AggressiveEarlyStopping(
        patience=5,
        min_delta=1e-4,
        save_path=config['save_path'],
        verbose=False
    )

    SUBGRAPH_BATCH_SIZE = config['batch_size']

    print(f"[Info] Start Training Refinement GNN ({config['epochs']} epochs max) | Batch Size: {SUBGRAPH_BATCH_SIZE}")

    for epoch in range(config['epochs']):

        avg_train_loss = run_one_epoch_subgraph(
            model, train_dataset, optimizer, mse_loss_fn, sim_loss_fn, comm_loss_fn,
            config['device'], is_training=True, batch_size=SUBGRAPH_BATCH_SIZE
        )

        avg_val_loss = run_one_epoch_subgraph(
            model, val_dataset, None, mse_loss_fn, sim_loss_fn, comm_loss_fn,
            config['device'], is_training=False, batch_size=SUBGRAPH_BATCH_SIZE
        )

        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("[Info] Early stopping triggered.")
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Graph Coarsening and Embedding Pipeline")
    parser.add_argument('--dataset', type=str, default='math',
                        help='Name of the dataset (e.g., math, stack, askubuntu)')

    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Subgraph Batch size (default: 2048)')

    args = parser.parse_args()

    RUN_ABLATION_NO_COARSENING = False
    dataset_name = args.dataset

    ABLATION_SUFFIX = "_ABLATION_NO_COARSE" if RUN_ABLATION_NO_COARSENING else ""
    P1_FILENAME = f"{dataset_name}_phase1_data_splits{ABLATION_SUFFIX}.pkl"
    P2_COARSE_FILENAME = f"{dataset_name}_phase2_embedding_splits_COARSE{ABLATION_SUFFIX}.pkl"
    P2_GOLDEN_FILENAME = f"{dataset_name}_phase2_embedding_splits_GOLDEN{ABLATION_SUFFIX}.pkl"

    config = {
        'dataset_name': dataset_name,
        'epochs': 100,
        'batch_size': args.batch_size,
        'lr': 3e-4,
        'patience': 7,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'p1_path': f"./saved_models/{P1_FILENAME}",
        'p2_coarse_path': f"./saved_models/{P2_COARSE_FILENAME}",
        'p2_golden_path': f"./saved_models/{P2_GOLDEN_FILENAME}",
        'save_path': f"saved_models/refinement_gnn_v15_1_{dataset_name}.pth"
    }

    print(f"--- Start Refinement GNN Training (BatchSize: {args.batch_size}) ---")
    try:
        train_refinement_model_v15(config)
    except Exception as e:
        print(f"\n[Critical Error] Uncaught Exception: {e}")
        import traceback
        traceback.print_exc()
