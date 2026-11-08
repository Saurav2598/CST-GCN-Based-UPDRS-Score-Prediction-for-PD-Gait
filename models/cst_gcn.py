import math

import torch
from torch import nn


class CSTGCNConv(nn.Module):
    """
    Cross-Spatiotemporal Graph Convolution.

    Input:
        x: (B, T, N, Cin)

    Adjacency:
        A: (K, N, Gamma*N)

    Output:
        (B, T, N, Cout)
    """

    def __init__(self, in_channels, out_channels, adjacency):
        super().__init__()

        adjacency = torch.as_tensor(
            adjacency, dtype=torch.float32
        )

        if adjacency.ndim != 3:
            raise ValueError(
                "adjacency must have shape (K, N, Gamma*N)"
            )

        num_subsets, num_nodes, num_sources = adjacency.shape

        if num_sources % num_nodes != 0:
            raise ValueError(
                "Source dimension must equal Gamma * N"
            )

        window_size = num_sources // num_nodes

        if window_size < 1 or window_size % 2 != 1:
            raise ValueError(
                "Gamma must be a positive odd integer"
            )

        self.num_nodes = num_nodes
        self.window_size = window_size

        # Fixed adjacency matrices.
        # These are not trainable parameters.
        self.register_buffer(
            "adjacency",
            adjacency.clone()
        )

        # One independently learnable matrix for every CST subset.
        self.weight = nn.Parameter(
            torch.empty(
                num_subsets,
                in_channels,
                out_channels
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(out_channels)
        )

        self.reset_parameters()

    def reset_parameters(self):

        with torch.no_grad():

            for weight_k in self.weight:
                nn.init.xavier_uniform_(weight_k)

            # Account for summing contributions
            # across multiple subsets.
            self.weight.mul_(
                1 / math.sqrt(self.weight.shape[0])
            )

            nn.init.zeros_(self.bias)

    def forward(self, x):

        if x.ndim != 4 or x.shape[2] != self.num_nodes:
            raise ValueError(
                f"x must have shape (B, T, {self.num_nodes}, Cin)"
            )

        if x.shape[-1] != self.weight.shape[1]:
            raise ValueError(
                f"Expected {self.weight.shape[1]} input channels"
            )

        B, T, N, Cin = x.shape

        half = self.window_size // 2

        # STEP 1: Temporal padding
        if half:

            pad = x.new_zeros(
                (B, half, N, Cin)
            )

            padded = torch.cat(
                (pad, x, pad),
                dim=1
            )

        else:
            padded = x

        # STEP 2: Construct temporal sliding windows
        windows = torch.stack(
            [
                padded[:, offset:offset + T]
                for offset in range(self.window_size)
            ],
            dim=2
        )

        # (B, T, Gamma, N, Cin)
        #          ↓
        # (B, T, Gamma*N, Cin)

        windows = windows.reshape(
            B,
            T,
            self.window_size * N,
            Cin
        )

        # STEP 3: Graph aggregation
        aggregated = torch.einsum(
            "knm,btmc->btknc",
            self.adjacency,
            windows
        )

        # STEP 4: Apply subset-specific weights
        # and sum over all subsets
        out = torch.einsum(
            "btknc,kco->btno",
            aggregated,
            self.weight
        )

        return out + self.bias


class CSTBlock(nn.Module):
    """
    CST-GCN convolution with normalization,
    residual connection, and activation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        adjacency,
        dropout=0.2
    ):
        super().__init__()

        self.conv = CSTGCNConv(
            in_channels,
            out_channels,
            adjacency
        )

        self.norm = nn.BatchNorm2d(
            out_channels
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(
                in_channels,
                out_channels,
                bias=False
            )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        residual = self.shortcut(x)

        # CST-GCN convolution
        y = self.conv(x)

        # BatchNorm2d expects (B, C, T, N)
        y = y.permute(0, 3, 1, 2)

        y = self.norm(y)

        # Restore (B, T, N, C)
        y = y.permute(0, 2, 3, 1)

        # Residual addition and activation
        y = self.activation(y + residual)

        return self.dropout(y)