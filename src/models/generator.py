"""HiFi-GAN Generator."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        # GroupNorm: stable at batch_size=1 unlike BatchNorm1d
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2
        )
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(F.leaky_relu(self.conv1(x), 0.2))
        x = self.norm2(F.leaky_relu(self.conv2(x), 0.2))
        return x + residual


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN style generator with progressive upsampling.

    Total upsampling: 8 × 5 × 4 × 2 = 320
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        upsample_rates: list | None = None,
        upsample_kernel_sizes: list | None = None,
        resblock_kernel_sizes: list | None = None,
        resblock_dilations: list | None = None,
    ):
        super().__init__()

        upsample_rates = upsample_rates or [8, 5, 4, 2]
        upsample_kernel_sizes = upsample_kernel_sizes or [16, 10, 8, 4]
        resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        resblock_dilations = resblock_dilations or [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.input_conv = nn.Conv1d(hidden_dim, 512, kernel_size=7, padding=3)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        channels = 512
        for rate, kernel in zip(upsample_rates, upsample_kernel_sizes, strict=True):
            out_channels = channels // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                )
            )
            resblock_list = nn.ModuleList()
            for k_size, dilations in zip(resblock_kernel_sizes, resblock_dilations, strict=True):
                for dil in dilations:
                    resblock_list.append(ResBlock(out_channels, k_size, dil))
            self.resblocks.append(resblock_list)
            channels = out_channels

        self.output_conv = nn.Conv1d(channels, 1, kernel_size=7, padding=3)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.input_conv(x), 0.2)

        for up, resblocks in zip(self.ups, self.resblocks, strict=True):
            x = F.leaky_relu(up(x), 0.2)
            xs = sum(rb(x) for rb in resblocks) / len(resblocks)
            x = xs

        x = self.output_conv(x)
        peak = x.abs().max(dim=-1, keepdim=True)[0]
        return x / (peak + 1e-8) * 0.95

