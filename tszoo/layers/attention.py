# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Chronos-2; see ../models/chronos2/NOTICE.md and LICENSE.
"""Multi-head attention with optional rotary positions and explicit masks."""

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(
        self,
        d_model,
        d_kv,
        num_heads,
        dropout_rate=0.0,
        rope_theta=10000.0,
        rotary=False,
    ):
        super().__init__()
        self.d_kv, self.num_heads = d_kv, num_heads
        self.dropout_rate, self.rope_theta, self.rotary = (
            dropout_rate,
            rope_theta,
            rotary,
        )
        width = num_heads * d_kv
        self.q = nn.Linear(d_model, width, bias=False)
        self.k = nn.Linear(d_model, width, bias=False)
        self.v = nn.Linear(d_model, width, bias=False)
        self.o = nn.Linear(width, d_model, bias=False)

    def forward(self, x, mask):
        c = self
        b, t, _ = x.shape
        q, k, v = [
            layer(x).reshape(b, t, c.num_heads, c.d_kv).transpose(1, 2)
            for layer in (self.q, self.k, self.v)
        ]
        if self.rotary:
            with torch.autocast(device_type=x.device.type, enabled=False):
                inv = 1 / (
                    c.rope_theta
                    ** (torch.arange(0, c.d_kv, 2, device=x.device).float() / c.d_kv)
                )
                freq = torch.arange(t, device=x.device).float()[:, None] * inv[None, :]
                phase = torch.cat([freq, freq], -1)
                cos, sin = phase.cos().to(q.dtype), phase.sin().to(q.dtype)

            def rotate(y):
                left, right = y.chunk(2, -1)
                return y * cos + torch.cat([-right, left], -1) * sin

            q, k = rotate(q), rotate(k)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=c.dropout_rate if self.training else 0.0,
            scale=1.0,
        )
        return self.o(y.transpose(1, 2).reshape(b, t, -1))
