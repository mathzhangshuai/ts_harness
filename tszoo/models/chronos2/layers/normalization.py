# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Chronos-2; see ../models/chronos2/NOTICE.md and LICENSE.
"""RMS normalization over the last dimension."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        y = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * y.to(self.weight.dtype)
