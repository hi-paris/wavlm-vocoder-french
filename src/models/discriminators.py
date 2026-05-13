"""HiFi-GAN Discriminators: MPD (Multi-Period) + MSD (Multi-Scale)

Basé sur le papier HiFi-GAN (Kong et al., 2020) - https://arxiv.org/abs/2010.05646

Usage:
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    # Forward
    mpd_real = mpd(real_audio)  # List of (output, fmaps)
    mpd_fake = mpd(fake_audio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class PeriodDiscriminator(nn.Module):
    """Discriminateur pour une période spécifique.
    
    Reshape la waveform 1D en 2D (batch, 1, T/period, period)
    puis applique des Conv2D.
    """
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (kernel_size//2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (kernel_size//2, 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (kernel_size//2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (kernel_size//2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] ou [B, T]
        Returns:
            output: [B, N] discriminator output
            fmaps: list of feature maps pour feature matching loss
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
        
        fmap = []
        b, c, t = x.shape
        
        # Pad pour que T soit divisible par period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        
        # Reshape: [B, 1, T] -> [B, 1, T/period, period]
        x = x.view(b, c, t // self.period, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD).
    
    Utilise 5 périodes différentes (2, 3, 5, 7, 11) pour capturer
    les structures périodiques à différentes fréquences.
    """
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x):
        """
        Args:
            x: [B, T] ou [B, 1, T]
        Returns:
            List of (output, fmaps) pour chaque période
        """
        ret = []
        for d in self.discriminators:
            out, fmap = d(x)
            ret.append((out, fmap))
        return ret


class ScaleDiscriminator(nn.Module):
    """Discriminateur pour une échelle (résolution temporelle).
    
    Utilise des Conv1D avec différents strides pour analyser
    la waveform à différentes échelles.
    """
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, 7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            norm_f(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            norm_f(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T]
        Returns:
            output: [B, N] discriminator output
            fmaps: list of feature maps
        """
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator (MSD).
    
    Utilise 3 discriminateurs à différentes échelles temporelles.
    Les échelles 2 et 3 utilisent un pooling progressif.
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),  # Échelle 1: spectral norm
            ScaleDiscriminator(),                         # Échelle 2: weight norm
            ScaleDiscriminator(),                         # Échelle 3: weight norm
        ])
        # Pooling pour réduire la résolution entre les échelles
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x):
        """
        Args:
            x: [B, T] ou [B, 1, T]
        Returns:
            List of (output, fmaps) pour chaque échelle
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
        
        ret = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i-1](x)
            out, fmap = d(x)
            ret.append((out, fmap))
        return ret


def get_discriminators(device='cuda'):
    """Factory function pour créer MPD + MSD."""
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    return mpd, msd


def count_parameters(model):
    """Compte le nombre de paramètres d'un modèle."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Test
    print("="*60)
    print("Test des discriminateurs HiFi-GAN")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    print(f"\nMPD parameters: {count_parameters(mpd):,}")
    print(f"MSD parameters: {count_parameters(msd):,}")
    print(f"Total discriminator parameters: {count_parameters(mpd) + count_parameters(msd):,}")
    
    # Test forward
    x = torch.randn(4, 32000, device=device)  # [B, T]
    
    print(f"\nInput shape: {x.shape}")
    
    mpd_out = mpd(x)
    print(f"\nMPD output: {len(mpd_out)} discriminators")
    for i, (out, fmaps) in enumerate(mpd_out):
        print(f"  Period {[2,3,5,7,11][i]}: output={out.shape}, fmaps={len(fmaps)}")
    
    msd_out = msd(x)
    print(f"\nMSD output: {len(msd_out)} discriminators")
    for i, (out, fmaps) in enumerate(msd_out):
        print(f"  Scale {i+1}: output={out.shape}, fmaps={len(fmaps)}")
    
    print("\n✅ Test réussi!")
