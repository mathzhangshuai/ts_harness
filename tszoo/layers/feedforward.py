# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Chronos-2; see ../models/chronos2/NOTICE.md and LICENSE.
"""ReLU feed-forward and residual projections."""

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, d_ff, dropout_rate=0.0):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, d_ff)
        self.output_layer = nn.Linear(d_ff, output_size)
        self.residual_layer = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(
            self.output_layer(F.relu(self.hidden_layer(x)))
        ) + self.residual_layer(x)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.0):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.wo(self.dropout(F.relu(self.wi(x))))
