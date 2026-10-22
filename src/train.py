import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import copy
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
import numpy as np

from dataset import TemporalEmbeddingDataset
from model import BlockDenoisingModelV14
from diffusion import Diffusion

def custom_collate_fn(batch):
    max_k = max(item['history_embeddings'].shape[0] for item in batch)
    embed_dim = batch[0]['history_embeddings'].shape[2]

    all_history_embs = []
    all_history_ts = []
    all_target_embs = []
    all_delta_ts = []
    all_edge_indices = []

    node_offset = 0

    for item in batch:
        hist_emb = item['history_embeddings']
        hist_rel_ts = item['history_relative_timestamps']
        target_emb = item['target_embedding']
        delta_t = item['delta_t']
        edge_index = item['edge_index']

        num_nodes_in_subgraph = target_emb.shape[0]
        k = hist_emb.shape[0]

        if k < max_k:
            padding_emb = torch.zeros((max_k - k, num_nodes_in_subgraph, embed_dim), dtype=hist_emb.dtype)
            hist_emb = torch.cat([hist_emb, padding_emb], dim=0)

            padding_ts = torch.full((max_k - k,), -99, dtype=hist_rel_ts.dtype)
            hist_rel_ts = torch.cat([hist_rel_ts, padding_ts], dim=0)

        hist_emb_transposed = hist_emb.permute(1, 0, 2)
        hist_rel_ts_expanded = hist_rel_ts.unsqueeze(0).expand(num_nodes_in_subgraph, -1)
        delta_t_expanded = delta_t.expand(num_nodes_in_subgraph)

        offset_edge_index = edge_index + node_offset

        all_history_embs.append(hist_emb_transposed)
        all_history_ts.append(hist_rel_ts_expanded)
        all_target_embs.append(target_emb)
        all_delta_ts.append(delta_t_expanded)
        all_edge_indices.append(offset_edge_index)

        node_offset += num_nodes_in_subgraph

    return {
        "history_embeddings": torch.cat(all_history_embs, dim=0),
        "history_relative_timestamps": torch.cat(all_history_ts, dim=0),
        "target_embedding": torch.cat(all_target_embs, dim=0),
        "delta_t": torch.cat(all_delta_ts, dim=0),
        "edge_index": torch.cat(all_edge_indices, dim=1)
    }

def bounded_contrastive_loss(prediction, target, temperature: float = 0.1, max_items: int = 4096):
    n = prediction.shape[0]
    if n > max_items:
        idx = torch.randperm(n, device=prediction.device)[:max_items]
        prediction = prediction.index_select(0, idx)
        target = target.index_select(0, idx)

    pred_norm = F.normalize(prediction, dim=1)
    target_norm = F.normalize(target, dim=1)
    logits = torch.mm(pred_norm, target_norm.t()) / temperature
    labels = torch.arange(logits.shape[0], device=prediction.device).long()
    return F.cross_entropy(logits, labels)

def validate_one_epoch(model: BlockDenoisingModelV14,
                       dataloader: DataLoader,
                       diffusion: Diffusion,
                       device: str,
                       evolution_loss_weight: float):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            target_embedding = batch["target_embedding"].to(device)
            history_embeddings = batch["history_embeddings"].to(device)
            history_relative_timestamps = batch["history_relative_timestamps"].to(device)
            delta_t = batch["delta_t"].to(device)
            edge_index = batch["edge_index"].to(device)

            t = torch.full((target_embedding.shape[0],), diffusion.timesteps // 2, device=device).long()
            noise = torch.randn_like(target_embedding)

            x_t = diffusion.q_sample(x_0=target_embedding, t=t, noise=noise)

            final_x0_prediction, evolution_guess = model(
                x_t, t,
                history_embeddings, history_relative_timestamps,
                delta_t,
                edge_index=edge_index
            )

            denoising_loss = F.mse_loss(final_x0_prediction, target_embedding)
            evolution_loss = F.l1_loss(evolution_guess, target_embedding)

            contrastive_loss = bounded_contrastive_loss(final_x0_prediction, target_embedding)

            total_loss_val = denoising_loss + evolution_loss_weight * evolution_loss + 0.1 * contrastive_loss

            total_val_loss += total_loss_val.item()

    if len(dataloader) == 0:
        return float('inf')
    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss

import joblib
import gc

def train(epochs, batch_size, learning_rate, device,
          p1_pkl_path: str,
          p2_coarse_pkl_path: str,
          model_save_path: str,
          dataset_name: str,
          forecast_horizons=(1,),
          evolution_loss_weight=0.1,
          early_stopping_patience=10):
    torch.set_num_threads(4)

    print(f"[Info] Init Diffusion Training Pipeline for {dataset_name}...")

    print(f"--- [Memory Opt] Preloading P1 data ({p1_pkl_path}) ---")
    try:
        full_p1 = joblib.load(p1_pkl_path, mmap_mode='r')
    except Exception as e:
        print(f"[Warning] mmap load failed ({e}), fallback to normal joblib.load")
        full_p1 = joblib.load(p1_pkl_path)

    train_p1_inputs = full_p1['train_aligned_inputs']
    val_p1_inputs = full_p1['val_aligned_inputs']
    num_nodes_global = full_p1['num_total_nodes']

    del full_p1
    gc.collect()
    print("--- [Memory Opt] P1 data preprocessing done, original dict released ---")

    print(f"--- [Memory Opt] Preloading P2 data ({p2_coarse_pkl_path}) ---")
    try:
        full_p2 = joblib.load(p2_coarse_pkl_path)
    except Exception:
        import pickle
        with open(p2_coarse_pkl_path, 'rb') as f:
            full_p2 = pickle.load(f)

    train_p2_raw = full_p2['train']
    val_p2_raw = full_p2['val']

    del full_p2
    gc.collect()
    print("--- [Memory Opt] P2 data preprocessing done, original dict released ---")

    print("[Dataset] Initializing Train Dataset with preloaded data...")
    train_dataset = TemporalEmbeddingDataset(
        p1_pkl_path=p1_pkl_path,
        p2_pkl_path=p2_coarse_pkl_path,
        dataset_name=dataset_name,
        split='train',
        samples_per_snapshot=2000,
        forecast_horizons=forecast_horizons,
        preloaded_inputs=train_p1_inputs,
        preloaded_p2_raw=train_p2_raw,
        preloaded_num_nodes=num_nodes_global
    )

    print("[Dataset] Initializing Val Dataset with preloaded data...")
    val_dataset = TemporalEmbeddingDataset(
        p1_pkl_path=p1_pkl_path,
        p2_pkl_path=p2_coarse_pkl_path,
        dataset_name=dataset_name,
        split='val',
        samples_per_snapshot=2000,
        forecast_horizons=forecast_horizons,
        preloaded_inputs=val_p1_inputs,
        preloaded_p2_raw=val_p2_raw,
        preloaded_num_nodes=num_nodes_global
    )

    del train_p1_inputs, val_p1_inputs, train_p2_raw, val_p2_raw
    gc.collect()
    print("--- [Memory Opt] External preloaded data cleaned, entering training loop ---")

    input_dim = 32
    try:
        if len(train_dataset) > 0:
            sample_item = train_dataset[0]
            input_dim = sample_item['target_embedding'].shape[-1]
    except Exception as e:
        print(f"[Warning] Dim detection failed, using {input_dim}: {e}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn
    )

    diffusion = Diffusion(timesteps=1000, device=device)

    model = BlockDenoisingModelV14(
        embed_dim=input_dim,
        time_embed_dim=32,
        n_layers=4,
        heads=4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=False)

    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_save_path = f"{model_save_path}_{dataset_name}.pth"

    dir_name = os.path.dirname(best_model_save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f"[Info] Best model will be saved to: {best_model_save_path}")
    print(">>> Training Mode: V14 Spatio-Temporal [Contrastive + CFG + Early Stop] <<<")

    print("\n--- Start Training Loop ---")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=True)

        for batch in train_dataloader:
            optimizer.zero_grad()

            target_embedding = batch["target_embedding"].to(device)
            history_embeddings = batch["history_embeddings"].to(device)
            history_relative_timestamps = batch["history_relative_timestamps"].to(device)
            delta_t = batch["delta_t"].to(device)
            edge_index = batch["edge_index"].to(device)

            with torch.no_grad():
                _, evolution_guess = model(
                    target_embedding,
                    torch.zeros_like(delta_t),
                    history_embeddings,
                    history_relative_timestamps,
                    delta_t,
                    edge_index=edge_index
                )
            x_c_real = evolution_guess.detach()

            if np.random.rand() < 0.1:
                model_input_hist = torch.zeros_like(history_embeddings)
                model_input_xc = torch.zeros_like(x_c_real)
            else:
                model_input_hist = history_embeddings
                model_input_xc = x_c_real

            t_norm = torch.rand(target_embedding.shape[0], device=device)
            t_distorted_norm = 2 * t_norm - t_norm.pow(2)
            t = (t_distorted_norm * diffusion.timesteps).long().clamp(0, diffusion.timesteps - 1)

            noise = torch.randn_like(target_embedding)

            x_t = diffusion.q_sample_coarse_target(
                x_0=target_embedding, t=t, x_c=x_c_real, noise=noise
            )

            if np.random.rand() < 0.9:
                with torch.no_grad():
                    x0_est, _ = model(
                        x_t, t,
                        model_input_hist, history_relative_timestamps,
                        delta_t,
                        x_c=model_input_xc,
                        edge_index=edge_index,
                        x_self=None,
                    )
                x_self = diffusion.q_sample_coarse_target(
                    x_0=x0_est.detach(), t=t, x_c=x_c_real
                )
            else:
                x_self = None

            final_x0_prediction, evolution_guess_for_loss = model(
                x_t, t,
                model_input_hist, history_relative_timestamps,
                delta_t,
                x_c=model_input_xc,
                edge_index=edge_index,
                x_self=x_self,
            )

            denoising_loss = F.mse_loss(final_x0_prediction, target_embedding)
            evolution_loss = F.l1_loss(evolution_guess_for_loss, target_embedding)

            loss_contrastive = bounded_contrastive_loss(final_x0_prediction, target_embedding)

            contrastive_weight = 0.1
            total_loss_val = denoising_loss + \
                             evolution_loss_weight * evolution_loss + \
                             contrastive_weight * loss_contrastive

            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += total_loss_val.item()

            del target_embedding, history_embeddings, edge_index, final_x0_prediction

        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else float('inf')

        avg_val_loss = validate_one_epoch(
            model, val_dataloader, diffusion, device, evolution_loss_weight
        )

        if avg_val_loss == float('inf'):
            print(f"[Warning] Val dataloader is empty. Using train loss for scheduling/early stopping.")
            avg_val_loss = avg_train_loss

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_save_path)
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stopping_patience:
            print(f"[Info] Early stopping triggered at epoch {epoch+1}.")
            try:
                model.load_state_dict(torch.load(best_model_save_path))
            except Exception as e:
                print(f"[Error] Failed to reload best model: {e}")
            break

    if early_stop_counter > 0 and early_stop_counter < early_stopping_patience:
        print(f"[Info] Training finished. Reloading best model...")
        try:
             model.load_state_dict(torch.load(best_model_save_path))
        except Exception as e:
             print(f"[Error] Failed to reload best model: {e}")

    return best_val_loss

import argparse

if __name__ == '__main__':
    EPOCHS = 20
    parser = argparse.ArgumentParser(description="Graph Coarsening and Embedding Pipeline")
    parser.add_argument('--dataset', type=str, default='math',
                        help='Name of the dataset (e.g., math, stack, askubuntu)')

    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for training (default: 2048)')
    parser.add_argument('--horizons', default='1',
                        help='Comma-separated direct forecast horizons')

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    FORECAST_HORIZONS = tuple(int(value) for value in args.horizons.split(',') if value)

    LEARNING_RATE = 5e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset

    RUN_ABLATION_NO_COARSENING = False
    ABLATION_SUFFIX = "_ABLATION_NO_COARSE" if RUN_ABLATION_NO_COARSENING else ""
    P1_FILENAME = f"{dataset_name}_phase1_data_splits{ABLATION_SUFFIX}.pkl"
    P2_FILENAME = f"{dataset_name}_phase2_embedding_splits_COARSE{ABLATION_SUFFIX}.pkl"
    PKL_FILE_PATH_PHASE1 = f"saved_models/{P1_FILENAME}"
    PKL_FILE_PATH_PHASE2 = f"saved_models/{P2_FILENAME}"
    MODEL_SAVE_PATH = "saved_models/diffusion_model_v14_gnn"

    print(f"--- Start Diffusion Training (Device: {DEVICE}, Batch Size: {BATCH_SIZE}) ---")

    train(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        p1_pkl_path=PKL_FILE_PATH_PHASE1,
        p2_coarse_pkl_path=PKL_FILE_PATH_PHASE2,
        model_save_path=MODEL_SAVE_PATH,
        dataset_name=dataset_name,
        forecast_horizons=FORECAST_HORIZONS,
    )
