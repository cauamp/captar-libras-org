import math
import torch
import torch.nn as nn


def sinusoid_encoding(length, dim, device):
    pos = torch.arange(length, dtype=torch.float, device=device)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CrossViewAttention(nn.Module):
    """Frontal frame features (query) attend to auxiliary streams (key/value): other views or depth.

    The residual is gated by tanh(gate) with gate initialised at 0, so at start the block is the identity
    and the model behaves like the frontal-only baseline.
    """

    def __init__(self, dim, aux_dim, num_views=1, num_heads=8, dropout=0.1):
        super(CrossViewAttention, self).__init__()
        self.aux_proj = nn.Linear(aux_dim, dim)
        self.view_embed = nn.Parameter(torch.zeros(num_views, 1, 1, dim))
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, query, aux, aux_len):
        """
        Args:
            - query: (B, C, T) frontal framewise features
            - aux: (B, V, C_aux, T_aux) auxiliary framewise features
            - aux_len: (B) valid length of the auxiliary streams
        Returns:
            - (B, C, T) fused features
        """
        q = query.permute(2, 0, 1)  # T, B, C
        batch, views, _, aux_temp = aux.shape
        kv = self.aux_proj(aux.permute(1, 3, 0, 2)) + self.view_embed  # V, T_aux, B, C
        kv = self.norm_kv(kv.reshape(views * aux_temp, batch, -1))
        q_pe = sinusoid_encoding(q.size(0), q.size(2), q.device)[:, None]
        kv_pe = sinusoid_encoding(aux_temp, q.size(2), q.device).repeat(views, 1)[:, None]
        # True marks padded positions; sequence index is v * T_aux + t
        mask = torch.arange(aux_temp, device=aux.device)[None] >= aux_len.to(aux.device)[:, None]
        mask = mask.repeat(1, views)
        out, _ = self.attn(self.norm_q(q) + q_pe, kv + kv_pe, kv, key_padding_mask=mask)
        q = q + torch.tanh(self.gate) * out
        return q.permute(1, 2, 0)
