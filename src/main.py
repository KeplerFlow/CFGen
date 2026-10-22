from coarsener import IncrementalCoarsener, static_coarsen
from cost_functions import AConvMatchCost
from candidate_strategies import EmbeddingCandidateStrategy
import sys
import os
import joblib

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from light_model import SpatioTemporalGCN
from embedder import TemporalEmbedder

from typing import List, Tuple, Dict
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling

from validate import run_coarsener_validation

import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm
import logging
import copy
import pickle
import random
import time
import pandas as pd
import gc
import argparse

def split_edges(data, val_ratio=0.1, test_ratio=0.0):
    transform = RandomLinkSplit(
        is_undirected=True,
        num_val=val_ratio,
        num_test=test_ratio,
        add_negative_train_samples=False,
        neg_sampling_ratio=0.0
    )

    train_data, val_data, test_data = transform(data)

    hidden_edges = val_data.edge_label_index
    if test_ratio > 0:
        hidden_edges = torch.cat([hidden_edges, test_data.edge_label_index], dim=1)

    return Data(
        num_nodes=data.num_nodes,
        visible_edge_index=train_data.edge_index,
        hidden_edge_index=hidden_edges,
        timestamp=data.timestamp
    )

def split_data_sequentially(all_snapshots: List[Tuple[int, Data]],
                            train_ratio=0.7, val_ratio=0.15,
                            lp_val_ratio=0.1) -> Dict[str, List[Data]]:
    num_total_snapshots = len(all_snapshots)
    num_train = int(num_total_snapshots * train_ratio)
    num_val = int(num_total_snapshots * val_ratio)

    splits = {}

    logging.info(f"Splitting {num_total_snapshots} snapshots...")

    train_snapshots = all_snapshots[:num_train]
    val_snapshots = all_snapshots[num_train : num_train + num_val]
    test_snapshots = all_snapshots[num_train + num_val:]

    logging.info(f"  Train split: {len(train_snapshots)} snapshots")
    logging.info(f"  Val split: {len(val_snapshots)} snapshots")
    logging.info(f"  Test split: {len(test_snapshots)} snapshots")

    logging.info(f"Splitting internal edges for train/val splits (lp_val_ratio={lp_val_ratio})...")

    splits['train'] = []
    for (timestamp, graph) in tqdm(train_snapshots, desc="  - Processing train snapshots", disable=True):
        graph.timestamp = timestamp
        num_edges = graph.num_edges
        if num_edges < 4:
            splits['train'].append(Data(
                num_nodes=graph.num_nodes,
                visible_edge_index=graph.edge_index,
                hidden_edge_index=torch.empty((2, 0), dtype=torch.long),
                timestamp=timestamp
            ))
        else:
            split_graph = split_edges(graph, val_ratio=lp_val_ratio, test_ratio=0.0)
            splits['train'].append(split_graph)

    splits['val'] = []
    for (timestamp, graph) in tqdm(val_snapshots, desc="  - Processing val snapshots", disable=True):
        graph.timestamp = timestamp
        num_edges = graph.num_edges
        if num_edges < 4:
            splits['val'].append(Data(
                num_nodes=graph.num_nodes,
                visible_edge_index=graph.edge_index,
                hidden_edge_index=torch.empty((2, 0), dtype=torch.long),
                timestamp=timestamp
            ))
        else:
            split_graph = split_edges(graph, val_ratio=lp_val_ratio, test_ratio=0.0)
            splits['val'].append(split_graph)

    splits['test'] = []
    for (timestamp, graph) in tqdm(test_snapshots, desc="  - Processing test snapshots", disable=True):
        splits['test'].append(Data(
            num_nodes=graph.num_nodes,
            visible_edge_index=graph.edge_index,
            hidden_edge_index=torch.empty((2, 0), dtype=torch.long),
            timestamp=timestamp
        ))

    logging.info("Data splitting done.")
    return splits

def load_temporal_graph(data_dir):
    print(f"--- [Ultra-Fast Load] Loading temporal graph data from directory '{data_dir}'... ---")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Error: directory not found '{data_dir}'.")

    files = sorted(os.listdir(data_dir), key=lambda f: int(f.split('.')[0]) if f[0].isdigit() else float('inf'))

    snapshots = []
    max_node_id = -1

    for filename in tqdm(files, desc="Loading Snapshots"):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath, sep=r'\s+', header=None, usecols=[0, 1],
                                 names=['u', 'v'], dtype={'u': 'int32', 'v': 'int32'}, comment=chr(35))
                df.dropna(inplace=True)

                if df.empty: continue

                edges_np = df.values

                current_max = edges_np.max()
                if current_max > max_node_id: max_node_id = current_max

                edge_index = torch.from_numpy(edges_np.T).long()

                pyg_graph = Data(edge_index=edge_index, num_nodes=None)
                ts = int(filename.split('.')[0])
                snapshots.append((ts, pyg_graph))

                del df, edges_np

            except Exception as e:
                logging.warning(f"Error loading {filename}: {e}")
                continue

    if max_node_id == -1:
        return [], 0

    num_total_nodes = int(max_node_id + 1)
    print(f"Snapshot loading done. Total nodes: {num_total_nodes}.")

    for _, graph in snapshots:
        graph.num_nodes = num_total_nodes

    return snapshots, num_total_nodes

def _build_coarse_adj_from_fine(fine_edge_index, mapping: np.ndarray,
                                num_supernodes: int):
    if fine_edge_index is None:
        return sp.csr_matrix((num_supernodes, num_supernodes), dtype=np.float32)
    if hasattr(fine_edge_index, 'cpu'):
        fine_edge_index = fine_edge_index.cpu().numpy()
    fe = np.asarray(fine_edge_index)
    if fe.ndim != 2 or fe.shape[0] != 2 or fe.shape[1] == 0:
        return sp.csr_matrix((num_supernodes, num_supernodes), dtype=np.float32)
    src = fe[0].astype(np.int64)
    dst = fe[1].astype(np.int64)
    valid = (src < mapping.shape[0]) & (dst < mapping.shape[0]) & (src >= 0) & (dst >= 0)
    src, dst = src[valid], dst[valid]
    data = np.ones(len(src), dtype=np.float32)
    return sp.coo_matrix((data, (mapping[src], mapping[dst])),
                         shape=(num_supernodes, num_supernodes)).tocsr()

def run_incremental_coarsening_on_split(split_name: str,
                                        snapshot_split: List[Data],
                                        coarsener: IncrementalCoarsener,
                                        current_visible_edges: set,
                                        num_total_nodes: int,
                                        initial_features: np.ndarray,
                                        timers: dict) -> Tuple[List[Dict], set, List[float]]:
    logging.info(f"\n--- Coarsening '{split_name}' split ---")

    if split_name == 'test':
        logging.info(f"  -> Test split detected, skipping coarsening computation (timestamps only).")
        dummy_test_data = [{'timestamp': g.timestamp} for g in snapshot_split]
        return dummy_test_data, current_visible_edges, []

    coarse_data_list = []
    compression_ratios_list = []

    initial_active_nodes = np.count_nonzero(coarsener.coarse_graph.supernode_sizes)
    if initial_active_nodes == 0: initial_active_nodes = 1

    original_proactive_k = coarsener.proactive_merge_k
    original_merge_threshold = coarsener.merge_threshold

    pbar = tqdm(snapshot_split, desc=f"Inc Coarsen [{split_name}]", unit="snap")

    ref_mapping = np.asarray(coarsener.coarse_graph.node_to_supernode).copy()
    num_ref_supernodes = int(ref_mapping.max()) + 1 if ref_mapping.size > 0 else 1

    for i, split_graph in enumerate(pbar):

        next_visible_edges_pyg = split_graph.visible_edge_index

        coarse_adj_snap = _build_coarse_adj_from_fine(next_visible_edges_pyg,
                                                       ref_mapping, num_ref_supernodes)

        num_active_supernodes = num_ref_supernodes
        compression_ratio = 0.0
        if num_total_nodes > 0:
            compression_ratio = 1.0 - (num_active_supernodes / float(num_total_nodes))

        compression_ratios_list.append(compression_ratio)

        if i % 10 == 0:
            torch.cuda.empty_cache()

        pbar.set_postfix({
            "Active": f"{num_active_supernodes}",
            "Ratio": f"{compression_ratio:.2%}"
        })

        coarse_data_list.append({
            'timestamp': split_graph.timestamp,
            'adj': coarse_adj_snap,
            'node_to_supernode': ref_mapping.copy(),
            'hidden_edge_index': split_graph.hidden_edge_index,
            'compression_ratio': compression_ratio,
            'validation_auc': 0
        })

    coarsener.proactive_merge_k = original_proactive_k
    coarsener.merge_threshold = original_merge_threshold

    return coarse_data_list, current_visible_edges, compression_ratios_list

def re_align_features_for_phase2(all_coarse_data: Dict[str, List[Dict]],
                                 initial_features: np.ndarray,
                                 num_total_nodes: int,
                                 feature_dim: int):
    logging.info("\n--- Preparing final aligned GNN inputs for Phase 2 (Matrix Mode) ---")
    MAX_SUPERNODES = num_total_nodes
    phase2_gnn_inputs = {'train': [], 'val': [], 'test': []}

    initial_features = initial_features.astype(np.float32)

    for split_name in ['train', 'val', 'test']:
        for snapshot_data in tqdm(all_coarse_data[split_name], desc=f"Aligning {split_name} split data", disable=True):

            if 'adj' not in snapshot_data:
                phase2_gnn_inputs[split_name].append({
                    'timestamp': snapshot_data['timestamp']
                })
                continue

            aligned_adj = snapshot_data['adj']

            mapping = snapshot_data['node_to_supernode']

            num_supernodes = aligned_adj.shape[0]
            sizes = np.bincount(mapping, minlength=num_supernodes)

            safe_sizes = sizes.copy().astype(np.float32)
            safe_sizes[safe_sizes == 0] = 1.0
            weights = 1.0 / safe_sizes

            node_weights = weights[mapping]

            row_indices = mapping
            col_indices = np.arange(num_total_nodes)

            aggregation_mat = sp.coo_matrix(
                (node_weights, (row_indices, col_indices)),
                shape=(num_supernodes, num_total_nodes)
            )

            aligned_features = aggregation_mat @ initial_features

            if sp.issparse(aligned_features):
                aligned_features = aligned_features.toarray()

            phase2_gnn_inputs[split_name].append({
                'timestamp': snapshot_data['timestamp'],
                'features': aligned_features,
                'adj': aligned_adj
            })

    logging.info("Phase 2 inputs ready.")
    return phase2_gnn_inputs

def run_phase2_embedding(
        phase1_data: dict,
        all_snapshots_original: list,
        device: torch.device,
        output_file_coarse: str,
        output_file_golden: str
    ):
    import pickle
    import gc

    logging.info("[Phase 2] Starting embedding generation...")

    train_inputs = phase1_data['train_aligned_inputs']
    val_inputs = phase1_data['val_aligned_inputs']
    test_inputs = phase1_data['test_aligned_inputs']

    feature_dim = phase1_data['feature_dim']
    num_total_nodes = phase1_data['num_total_nodes']

    if 'global_sgc_features' in phase1_data:
        global_features = phase1_data['global_sgc_features']
    elif 'initial_features' in phase1_data:
        global_features = phase1_data['initial_features']
    else:
        logging.warning("Global features not found; initializing with zero features.")
        global_features = np.zeros((num_total_nodes, feature_dim), dtype=np.float32)

    train_timestamps = {snap['timestamp'] for snap in train_inputs}
    val_timestamps = {snap['timestamp'] for snap in val_inputs}

    all_aligned_inputs_coarse = sorted(train_inputs + val_inputs, key=lambda x: x['timestamp'])

    temporal_dim = 16
    hidden_channels = 64
    out_channels = 32

    gnn_model = SpatioTemporalGCN(
        feature_dim=feature_dim,
        temporal_dim=temporal_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    ).to(device)

    embedder = TemporalEmbedder(gnn_model, temporal_decay=0.8)
    phase2_val_loss = embedder.fit_link_prediction(
        train_inputs,
        val_inputs,
        epochs=8,
        snapshots_per_epoch=12,
        max_edges_per_snapshot=8000,
        learning_rate=1e-3,
        patience=3,
        seed=42,
    )
    logging.info(
        f"Phase-2 link-supervised encoder best validation loss: "
        f"{phase2_val_loss:.6f}"
    )

    coarse_emb_generator = embedder.generate_embedding_sequence(all_aligned_inputs_coarse)

    coarse_embedding_splits = {'train': [], 'val': [], 'test': []}

    for i, emb_data in enumerate(tqdm(coarse_emb_generator, total=len(all_aligned_inputs_coarse), desc="Exporting Coarse Embs")):

        ts = emb_data['timestamp']
        full_embedding = emb_data['embedding']

        full_embedding_np = full_embedding.numpy()

        compact_data = {
            'timestamp': ts,
            'embedding': full_embedding_np,
            'indices': None,
            'shape_hint': full_embedding_np.shape
        }

        if ts in train_timestamps:
            coarse_embedding_splits['train'].append(compact_data)
        elif ts in val_timestamps:
            coarse_embedding_splits['val'].append(compact_data)
        else:
            coarse_embedding_splits['test'].append(compact_data)

        del full_embedding, full_embedding_np, emb_data

        if i % 5 == 0:
            gc.collect()

    logging.info(f"Saving coarse embeddings to: {output_file_coarse}")
    try:
        with open(output_file_coarse, 'wb') as f:
            pickle.dump(coarse_embedding_splits, f, protocol=4)
        logging.info("Coarse embeddings saved successfully!")
    except Exception as e:
        logging.error(f"Failed to save coarse embeddings: {str(e)}")

    del coarse_embedding_splits
    gc.collect()

    all_original_inputs = []
    train_val_timestamps = train_timestamps.union(val_timestamps)

    for timestamp, graph in tqdm(all_snapshots_original, desc="Building original graph inputs", disable=False):
        if timestamp not in train_val_timestamps:
            continue

        adj_orig = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes).tocsr()

        if global_features.shape[0] >= graph.num_nodes:
            current_features = global_features[:graph.num_nodes]
        else:
            current_features = np.zeros((graph.num_nodes, feature_dim), dtype=np.float32)
            current_features[:global_features.shape[0]] = global_features

        all_original_inputs.append({
            'timestamp': timestamp,
            'features': current_features,
            'adj': adj_orig
        })

    golden_emb_sequence = embedder.generate_embedding_sequence(all_original_inputs)
    golden_embedding_splits = {'train': [], 'val': [], 'test': []}

    for i, emb_data in enumerate(tqdm(golden_emb_sequence, total=len(all_original_inputs), desc="Exporting Golden Embs", disable=False)):
        ts = emb_data['timestamp']
        full_embedding_np = emb_data['embedding'].numpy()

        adj_orig = all_original_inputs[i]['adj']
        active_mask = (adj_orig.getnnz(axis=1) > 0)
        active_indices = np.where(active_mask)[0].astype(np.int32)
        compact_embedding = full_embedding_np[active_indices]

        compact_data = {
            'timestamp': ts,
            'embedding': compact_embedding,
            'indices': active_indices,
            'shape_hint': full_embedding_np.shape
        }

        if ts in train_timestamps:
            golden_embedding_splits['train'].append(compact_data)
        elif ts in val_timestamps:
            golden_embedding_splits['val'].append(compact_data)

        del full_embedding_np, compact_embedding, emb_data
        if i % 5 == 0:
            gc.collect()

    for snap in test_inputs:
        golden_embedding_splits['test'].append({'timestamp': snap['timestamp']})

    logging.info(f"Saving Golden embeddings to: {output_file_golden}")
    with open(output_file_golden, 'wb') as f:
        pickle.dump(golden_embedding_splits, f, protocol=4)

    logging.info("Embedding generation done (Train/Val only)!")

    return {
        'coarse_embeddings': None,
        'golden_embeddings': golden_embedding_splits
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Graph Coarsening and Embedding Pipeline")
    parser.add_argument('--dataset', type=str, default='bitcoin',
                        help='Name of the dataset (e.g., math, stack, bitcoin)')
    parser.add_argument('--ratio', type=float, default=0.95, help='Target coarsening ratio')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    timers = {}
    dataset_name = args.dataset
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DUMMY_DATA_DIR = os.path.join(_REPO_ROOT, "datasets", dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    STATIC_TARGET_RATIO = 1.0 - args.ratio
    if STATIC_TARGET_RATIO < 0.01: STATIC_TARGET_RATIO = 0.01

    INCREMENTAL_MERGE_THRESHOLD = 50.0/2
    PROACTIVE_MERGE_K = 0

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    LP_VALIDATION_RATIO = 0.1

    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    OUTPUT_FILE_PHASE1_ALL = f"./saved_models/{dataset_name}_phase1_data_splits.pkl"
    OUTPUT_FILE_PHASE2_COARSE = f"./saved_models/{dataset_name}_phase2_embedding_splits_COARSE.pkl"
    OUTPUT_FILE_PHASE2_GOLDEN = f"./saved_models/{dataset_name}_phase2_embedding_splits_GOLDEN.pkl"

    RUN_PHASE_2_EMBEDDING = True
    all_compression_ratios = []

    all_snapshots, num_total_nodes = load_temporal_graph(DUMMY_DATA_DIR)

    if num_total_nodes == 0 or not all_snapshots:
        logging.error("Graph data is empty; cannot proceed.")
        sys.exit(1)

    snapshot_splits = split_data_sequentially(
        all_snapshots,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        lp_val_ratio=LP_VALIDATION_RATIO
    )

    feature_dim = 32
    current_visible_edges = set()

    all_coarse_data = {'train': [], 'val': [], 'test': []}

    logging.info("="*50 + "\n [Phase 1-A] Static coarsening...\n" + "="*50)
    first_train_graph = snapshot_splits['train'][0]

    train_static_edge_index = torch.cat(
        [g.visible_edge_index for g in snapshot_splits['train']],
        dim=1,
    )
    initial_static_adj_full = to_scipy_sparse_matrix(
        train_static_edge_index,
        num_nodes=num_total_nodes
    ).tocsr()

    identity_features = sp.identity(num_total_nodes, format='csr')
    cost_func = AConvMatchCost()

    candidate_strat = EmbeddingCandidateStrategy(
        identity_features,
        initial_static_adj_full,
        k_neighbors=20,
        sgc_k_hops=2,
        output_dim=feature_dim
    )
    initial_features = candidate_strat.original_sgc_embeddings

    logging.info(f" -> Running Static Coarsen (Target Ratio: {STATIC_TARGET_RATIO:.3f})")
    initial_visible_adj = to_scipy_sparse_matrix(
        first_train_graph.visible_edge_index,
        num_nodes=num_total_nodes
    ).tocsr()

    initial_coarse_graph = static_coarsen(
        initial_adj=initial_visible_adj,
        initial_features=initial_features,
        candidate_strategy=candidate_strat,
        cost_function=cost_func,
        target_coarsening_ratio=STATIC_TARGET_RATIO
    )

    num_active = np.count_nonzero(initial_coarse_graph.supernode_sizes)
    comp_ratio = 1.0 - (num_active / float(num_total_nodes))

    current_visible_edges = set(tuple(sorted(edge)) for edge in first_train_graph.visible_edge_index.t().tolist())

    t0_record = {
        'timestamp': first_train_graph.timestamp,
        'adj': _build_coarse_adj_from_fine(
            first_train_graph.visible_edge_index,
            np.asarray(initial_coarse_graph.node_to_supernode),
            int(np.asarray(initial_coarse_graph.node_to_supernode).max()) + 1),
        'supernode_map': copy.deepcopy(initial_coarse_graph.supernode_to_nodes),
        'node_to_supernode': initial_coarse_graph.node_to_supernode.copy(),
        'hidden_edge_index': first_train_graph.hidden_edge_index,
        'compression_ratio': comp_ratio,
        'validation_auc': 0
    }
    all_coarse_data['train'].append(t0_record)
    all_compression_ratios.append(comp_ratio)

    coarsener = IncrementalCoarsener(
        initial_coarse_graph=initial_coarse_graph,
        cost_function=AConvMatchCost(),
        candidate_strategy=EmbeddingCandidateStrategy(
            sp.identity(num_total_nodes, format='csr'),
            sp.csr_matrix((num_total_nodes, num_total_nodes)),
            output_dim=feature_dim
        ),
        merge_threshold=INCREMENTAL_MERGE_THRESHOLD,
        proactive_merge_k=PROACTIVE_MERGE_K
    )
    coarsener.set_target_ratio(STATIC_TARGET_RATIO)

    gc.collect()

    logging.info("="*50 + "\n [Phase 1-B] Train incremental...\n" + "="*50)
    split_data = snapshot_splits['train'][1:]
    inc_data, current_visible_edges, ratios = run_incremental_coarsening_on_split(
        'train', split_data, coarsener, current_visible_edges,
        num_total_nodes, initial_features, timers
    )
    all_coarse_data['train'].extend(inc_data)
    all_compression_ratios.extend(ratios)
    gc.collect()

    logging.info("="*50 + "\n [Phase 1-C] Val incremental...\n" + "="*50)
    split_data = snapshot_splits['val']
    inc_data, current_visible_edges, ratios = run_incremental_coarsening_on_split(
        'val', split_data, coarsener, current_visible_edges,
        num_total_nodes, initial_features, timers
    )
    all_coarse_data['val'].extend(inc_data)
    all_compression_ratios.extend(ratios)
    gc.collect()

    logging.info("="*50 + "\n [Phase 1-D] Test incremental...\n" + "="*50)
    split_data = snapshot_splits['test']
    inc_data, current_visible_edges, ratios = run_incremental_coarsening_on_split(
        'test', split_data, coarsener, current_visible_edges,
        num_total_nodes, initial_features, timers
    )
    all_coarse_data['test'].extend(inc_data)
    all_compression_ratios.extend(ratios)
    gc.collect()

    logging.info("="*50 + "\n [Phase 1-E] Final merge and save...\n" + "="*50)

    phase2_gnn_inputs = re_align_features_for_phase2(
        all_coarse_data, initial_features, num_total_nodes, feature_dim
    )

    train_history_list = [snap['node_to_supernode'] for snap in all_coarse_data['train']]
    val_history_list = [snap['node_to_supernode'] for snap in all_coarse_data['val']]
    test_history_list = [snap.get('node_to_supernode') for snap in all_coarse_data['test'] if 'node_to_supernode' in snap]

    data_to_save_phase1 = {
        'train_aligned_inputs': phase2_gnn_inputs['train'],
        'val_aligned_inputs': phase2_gnn_inputs['val'],
        'test_aligned_inputs': phase2_gnn_inputs['test'],

        'train_hidden_edges': [g.hidden_edge_index for g in snapshot_splits['train']],
        'val_hidden_edges': [g.hidden_edge_index for g in snapshot_splits['val']],

        'train_partition_history': np.vstack(train_history_list).astype(np.int32) if train_history_list else np.array([], dtype=np.int32),
        'val_partition_history': np.vstack(val_history_list).astype(np.int32) if val_history_list else np.array([], dtype=np.int32),
        'test_partition_history': np.vstack(test_history_list).astype(np.int32) if test_history_list else np.array([], dtype=np.int32),

        'num_total_nodes': num_total_nodes,
        'feature_dim': feature_dim,
        'initial_features': initial_features.astype(np.float32),
        'storage_mode': 'full'
    }

    del all_coarse_data, snapshot_splits, coarsener
    gc.collect()

    logging.info(f"Saving final merged result to {OUTPUT_FILE_PHASE1_ALL}...")
    joblib.dump(data_to_save_phase1, OUTPUT_FILE_PHASE1_ALL, compress=('gzip', 3))

    logging.info("Cleaning up Phase 1 memory, preparing to enter Phase 2...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Memory cleanup done.")

    if RUN_PHASE_2_EMBEDDING and data_to_save_phase1 is not None:
        run_phase2_embedding(
            phase1_data=data_to_save_phase1,
            all_snapshots_original=all_snapshots,
            device=device,
            output_file_coarse=OUTPUT_FILE_PHASE2_COARSE,
            output_file_golden=OUTPUT_FILE_PHASE2_GOLDEN
        )

    logging.info("Pipeline done.")
