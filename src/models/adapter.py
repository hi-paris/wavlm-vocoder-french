"""
WavLM Adapter and Layer Fusion
===============================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerFusion(nn.Module):
    """
    Fuse multiple WavLM layers into a single representation.

    Supports:
        - "average": Simple averaging
        - "learned": Learned weighted combination
    """

    def __init__(self, num_layers=12, fusion_type="learned"):
        super().__init__()

        self.num_layers = num_layers
        self.fusion_type = fusion_type

        if fusion_type == "learned":
            # Learnable weights (constrained to sum to 1)
            self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Tuple of (B, T, 768) tensors from WavLM

        Returns:
            (B, T, 768) fused representation
        """
        # Take last N layers
        layers = hidden_states[-self.num_layers :]

        if self.fusion_type == "average":
            # Simple averaging
            stacked = torch.stack(layers, dim=0)  # (N, B, T, 768)
            fused = stacked.mean(dim=0)  # (B, T, 768)

        elif self.fusion_type == "learned":
            # Learned weighted combination
            weights = F.softmax(self.weights, dim=0)  # Normalize to sum=1

            fused = 0
            for i, layer in enumerate(layers):
                fused = fused + weights[i] * layer

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        return fused


class WavLMAdapter(nn.Module):
    """
    Adapter to project WavLM features to generator input space.

    Architecture:
        (B, T, 768) → Linear → Conv blocks → (B, hidden_dim, T)
    """

    def __init__(self, wavlm_dim=768, hidden_dim=256, num_layers=3, kernel_size=7, dropout=0.1):
        super().__init__()

        # Linear projection
        self.input_proj = nn.Linear(wavlm_dim, hidden_dim)

        # Convolutional blocks with residual connections
        # GroupNorm instead of BatchNorm1d: stable at batch_size=1 (inference)
        self.conv_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict(
                {
                    "conv": nn.Conv1d(
                        hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2
                    ),
                    "norm": nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
                    "dropout": nn.Dropout(dropout),
                }
            )
            self.conv_blocks.append(block)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, T, 768)

        Returns:
            (B, hidden_dim, T)
        """
        # Linear projection
        x = self.input_proj(x)  # (B, T, hidden_dim)
        x = x.transpose(1, 2)  # (B, hidden_dim, T)

        # Conv blocks with residuals
        for block in self.conv_blocks:
            residual = x
            x = block["conv"](x)
            x = F.gelu(x)
            x = block["norm"](x)
            x = block["dropout"](x)
            x = x + residual

        return x

