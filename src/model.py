import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules import (
    SinusoidalPosEmb,
    TimeTransformerEncoder
)
from torch_geometric.nn import GATv2Conv

class BlockAttentionLayer(nn.Module):
    def __init__(self, embed_dim, heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.spatial_gnn = GATv2Conv(embed_dim, embed_dim, heads=heads, concat=False, dropout=dropout)
        self.norm_spatial = nn.LayerNorm(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self,
                x,
                history_sequence,
                key_padding_mask=None,
                edge_index=None
               ):

        attn_output, _ = self.cross_attn(
            query=x,
            key=history_sequence,
            value=history_sequence,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + attn_output)

        if edge_index is not None and edge_index.numel() > 0:
            x_squeezed = x.squeeze(1)

            gnn_output = self.spatial_gnn(x_squeezed, edge_index)

            x = self.norm_spatial(x + gnn_output.unsqueeze(1))

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class BlockDenoisingModelV14(nn.Module):
    def __init__(self, embed_dim: int, time_embed_dim: int, n_layers: int = 4, heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(embed_dim, embed_dim)

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim), nn.Linear(time_embed_dim, embed_dim), nn.Mish(), nn.Linear(embed_dim, embed_dim)
        )

        self.delta_t_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim), nn.Linear(time_embed_dim, embed_dim), nn.Mish(), nn.Linear(embed_dim, embed_dim)
        )

        self.history_time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.xc_proj = nn.Linear(embed_dim, embed_dim)

        self.history_summarizer = TimeTransformerEncoder(embed_dim=embed_dim, time_embed_dim=time_embed_dim)
        self.evolution_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.block_attention_layers = nn.ModuleList([
            BlockAttentionLayer(embed_dim, heads) for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                x_t,
                t,
                history_embeddings,
                history_relative_timestamps,
                delta_t,
                x_c=None,
                x_self=None,
                edge_index=None
               ):

        batch_size, seq_len, _ = history_embeddings.shape

        history_for_summarizer = history_embeddings

        history_summary_pooled = self.history_summarizer(history_for_summarizer, history_relative_timestamps)

        evolution_guess = self.evolution_predictor(history_summary_pooled)

        if x_c is None:
            x_c = evolution_guess.detach()

        key_padding_mask = (history_relative_timestamps == -99)

        if not key_padding_mask.any():
            key_padding_mask = None

        time_emb = self.history_time_encoder(history_relative_timestamps.reshape(batch_size * seq_len))
        time_emb = time_emb.view(batch_size, seq_len, self.embed_dim)

        enriched_history = history_embeddings + time_emb

        history_sequence = enriched_history

        if x_self is None:
            x_self = torch.zeros_like(x_t)
        h = self.input_proj(x_t + x_self)

        t_embed = self.time_encoder(t)
        delta_t_embed = self.delta_t_encoder(delta_t)
        combined_time_embed = t_embed + delta_t_embed

        h = h + combined_time_embed
        h = h + self.xc_proj(x_c)

        h = h.unsqueeze(1)

        for layer in self.block_attention_layers:
            h = layer(
                h,
                history_sequence,
                key_padding_mask=key_padding_mask,
                edge_index=edge_index
            )

        h = h.squeeze(1)

        final_x0_prediction = self.output_proj(h)

        return final_x0_prediction, evolution_guess
