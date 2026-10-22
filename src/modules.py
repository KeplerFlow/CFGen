import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        time = time.abs().float() + 1e-5
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, time_embed_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, history_embeddings, history_relative_timestamps):
        b, k, d = history_embeddings.shape

        time_emb = self.time_encoder(history_relative_timestamps.view(b * k))
        time_emb = time_emb.view(b, k, d)

        enriched_history = history_embeddings + time_emb

        padding_mask = (history_relative_timestamps == -99)

        if not padding_mask.any():
            padding_mask = None

        encoded_history = self.transformer_encoder(
            enriched_history,
            src_key_padding_mask=padding_mask
        )

        history_summary = encoded_history[:, -1, :]

        return history_summary
