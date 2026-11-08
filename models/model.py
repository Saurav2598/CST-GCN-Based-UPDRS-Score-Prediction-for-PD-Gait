import torch
from torch import nn

from graph.cst_graph import CSTGraph
from .cst_gcn import CSTBlock


class CSTGCNClassifier(nn.Module):

    def __init__(
        self,
        num_classes=4,
        spatial_radius=2,
        temporal_window=3,
        channels=(64, 64, 128),
        dropout=0.2
    ):
        super().__init__()

        # STEP 1: Build graph supports
        graph = CSTGraph(
            spatial_radius=spatial_radius,
            temporal_window=temporal_window
        )

        A_st = torch.from_numpy(
            graph.A_st_normalized
        )

        self.num_nodes = graph.num_nodes

        # STEP 2: Feature embedding
        self.embedding = nn.Linear(
            3,
            channels[0]
        )

        # STEP 3: Construct CST-GCN blocks
        blocks = []

        current_channels = channels[0]

        for output_channels in channels:

            blocks.append(
                CSTBlock(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    adjacency=A_st,
                    dropout=dropout
                )
            )

            current_channels = output_channels

        self.blocks = nn.ModuleList(blocks)

        # STEP 4: Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(current_channels),
            nn.Dropout(dropout),
            nn.Linear(
                current_channels,
                num_classes
            )
        )

    def forward(self, x, lengths=None):

        # Expected input: (B, T, 9, 3)
        if x.ndim != 4 or x.shape[2:] != (self.num_nodes, 3):
            raise ValueError(
                f"Expected x shape (B, T, {self.num_nodes}, 3)"
            )

        # Feature embedding
        x = self.embedding(x)

        # CST-GCN feature extraction
        for block in self.blocks:
            x = block(x)

        # Global average pooling across joints
        x = x.mean(dim=2)

        # Temporal pooling
        if lengths is None:

            pooled = x.mean(dim=1)

        else:

            lengths = torch.as_tensor(
                lengths,
                device=x.device
            )

            if (
                lengths.ndim != 1
                or lengths.shape[0] != x.shape[0]
                or (lengths < 1).any()
                or (lengths > x.shape[1]).any()
            ):
                raise ValueError(
                    "lengths must contain one valid length per sample"
                )

            mask = (
                torch.arange(
                    x.shape[1],
                    device=x.device
                )[None, :] < lengths[:, None]
            )

            pooled = (
                (x * mask.unsqueeze(-1)).sum(dim=1)
                / lengths.unsqueeze(-1)
            )

        # Output classification logits
        return self.classifier(pooled)