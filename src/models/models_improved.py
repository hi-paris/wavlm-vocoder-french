"""Modèles améliorés pour WavLM2Audio.

Améliorations par rapport à la version originale:
1. WeightNorm au lieu de BatchNorm
2. Weighted-sum des couches WavLM
3. Architecture ResBlock améliorée
4. Snake activation (optionnel, comme BigVGAN)

Usage:
    model = WavLM2AudioImproved(
        wavlm_model_name="microsoft/wavlm-base-plus",
        hidden_dim=256,
        use_weighted_layers=True,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import logging
import math

logger = logging.getLogger(__name__)


def init_weights(m, mean=0.0, std=0.01):
    """Initialisation des poids pour Conv1d."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class Snake(nn.Module):
    """Snake activation function (BigVGAN).
    
    snake(x) = x + (1/alpha) * sin²(alpha * x)
    
    Meilleure pour la modélisation des sinusoïdes (parole, musique).
    """
    def __init__(self, channels, alpha=1.0, alpha_logscale=False):
        super().__init__()
        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(1, channels, 1) + math.log(alpha))
            self.alpha_logscale = True
        else:
            self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha)
            self.alpha_logscale = False
    
    def forward(self, x):
        alpha = self.alpha.exp() if self.alpha_logscale else self.alpha
        return x + (1.0 / (alpha + 1e-8)) * torch.sin(alpha * x).pow(2)


class WavLMAdapterImproved(nn.Module):
    """Adapter amélioré: WeightNorm + Skip connections renforcées.
    
    Changements:
    - WeightNorm au lieu de BatchNorm
    - Meilleure initialisation
    - Activation configurable (GELU ou Snake)
    """
    def __init__(
        self,
        wavlm_dim=768,
        hidden_dim=256,
        num_layers=3,
        kernel_size=7,
        dropout=0.1,
        use_snake=False,
    ):
        super().__init__()
        logger.info(f"[WavLMAdapterImproved] {wavlm_dim}→{hidden_dim}, {num_layers} layers, WeightNorm")
        
        self.wavlm_dim = wavlm_dim
        self.hidden_dim = hidden_dim
        
        # Projection initiale
        self.input_proj = nn.Linear(wavlm_dim, hidden_dim)
        
        # Blocs convolutionnels avec WeightNorm
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.conv_blocks.append(
                nn.ModuleDict({
                    'conv': weight_norm(nn.Conv1d(
                        hidden_dim, hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )),
                    'dropout': nn.Dropout(dropout),
                })
            )
        
        # Activation
        self.use_snake = use_snake
        if use_snake:
            self.activations = nn.ModuleList([Snake(hidden_dim) for _ in range(num_layers)])
        
        self.apply(init_weights)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, 768) - WavLM features
        Returns:
            (B, hidden_dim, T) - Adapted features
        """
        # Projection
        x = self.input_proj(x)  # (B, T, hidden_dim)
        x = x.transpose(1, 2)   # (B, hidden_dim, T)
        
        # Conv blocks avec skip connections
        for i, block in enumerate(self.conv_blocks):
            residual = x
            x = block['conv'](x)
            if self.use_snake:
                x = self.activations[i](x)
            else:
                x = F.gelu(x)
            x = block['dropout'](x)
            x = x + residual
        
        return x
    
    def remove_weight_norm(self):
        """Enlève WeightNorm pour l'inférence."""
        for block in self.conv_blocks:
            remove_weight_norm(block['conv'])


class ResBlockImproved(nn.Module):
    """ResBlock avec WeightNorm (style HiFi-GAN).
    
    Changements:
    - WeightNorm au lieu de BatchNorm
    - LeakyReLU cohérent
    - Dilated convolutions
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), use_snake=False):
        super().__init__()
        self.use_snake = use_snake
        
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        
        if use_snake:
            self.activations1 = nn.ModuleList()
            self.activations2 = nn.ModuleList()
        
        for d in dilation:
            self.convs1.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=d, padding=(kernel_size * d - d) // 2
                ))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=1, padding=(kernel_size - 1) // 2
                ))
            )
            if use_snake:
                self.activations1.append(Snake(channels))
                self.activations2.append(Snake(channels))
        
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
    
    def forward(self, x):
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            xt = x
            if self.use_snake:
                xt = self.activations1[i](xt)
            else:
                xt = F.leaky_relu(xt, 0.1)
            xt = c1(xt)
            if self.use_snake:
                xt = self.activations2[i](xt)
            else:
                xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class HiFiGeneratorImproved(nn.Module):
    """Générateur HiFi-GAN amélioré avec WeightNorm.
    
    Changements:
    - WeightNorm partout
    - ResBlocks améliorés
    - Activation configurable (LeakyReLU ou Snake)
    - Upsampling rates configurables
    """
    def __init__(
        self,
        hidden_dim=256,
        upsample_rates=[8, 5, 4, 2],
        upsample_kernel_sizes=[16, 10, 8, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        use_snake=False,
    ):
        super().__init__()
        logger.info(f"[HiFiGeneratorImproved] hidden_dim={hidden_dim}, rates={upsample_rates}")
        
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.use_snake = use_snake
        
        # Conv d'entrée
        self.conv_pre = weight_norm(nn.Conv1d(hidden_dim, 512, 7, 1, 3))
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        channels = 512
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_channels = channels // 2
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    channels, out_channels, kernel,
                    stride=rate, padding=(kernel - rate) // 2
                ))
            )
            channels = out_channels
        
        # ResBlocks
        self.resblocks = nn.ModuleList()
        channels = 512
        for i in range(len(self.ups)):
            channels = channels // 2
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(ResBlockImproved(channels, k, d, use_snake=use_snake))
        
        # Activations pour upsampling
        if use_snake:
            self.up_activations = nn.ModuleList([Snake(512 // (2 ** (i+1))) for i in range(len(self.ups))])
        
        # Conv de sortie
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, 3))
        
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    
    def forward(self, x):
        """
        Args:
            x: (B, hidden_dim, T') - Features de l'adapter
        Returns:
            (B, 1, T) - Audio généré
        """
        x = self.conv_pre(x)
        
        for i, up in enumerate(self.ups):
            if self.use_snake:
                x = self.up_activations[i](x)
            else:
                x = F.leaky_relu(x, 0.1)
            x = up(x)
            
            # Appliquer les ResBlocks et moyenner
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels
        
        if self.use_snake:
            x = Snake(x.shape[1]).to(x.device)(x)
        else:
            x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class WavLM2AudioImproved(nn.Module):
    """Modèle complet amélioré: WavLM → Adapter → Generator.
    
    Améliorations:
    1. Weighted-sum des couches WavLM (pas seulement la dernière)
    2. WeightNorm dans tout le générateur
    3. Snake activation optionnelle (BigVGAN-style)
    """
    def __init__(
        self,
        wavlm_model_name="microsoft/wavlm-base-plus",
        hidden_dim=256,
        num_adapter_layers=3,
        kernel_size=7,
        freeze_wavlm=True,
        dropout=0.1,
        use_weighted_layers=True,  # NOUVEAU: weighted sum des couches WavLM
        use_snake=False,           # NOUVEAU: Snake activation
    ):
        super().__init__()
        
        logger.info("="*80)
        logger.info("[WavLM2AudioImproved] INITIALISATION")
        logger.info("="*80)
        logger.info(f"  wavlm_model_name: {wavlm_model_name}")
        logger.info(f"  hidden_dim: {hidden_dim}")
        logger.info(f"  use_weighted_layers: {use_weighted_layers}")
        logger.info(f"  use_snake: {use_snake}")
        
        self.freeze_wavlm = freeze_wavlm
        self.use_weighted_layers = use_weighted_layers
        
        # Charger WavLM
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                wavlm_model_name, local_files_only=True
            )
            self.wavlm = WavLMModel.from_pretrained(
                wavlm_model_name, local_files_only=True
            )
        except Exception:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_model_name)
            self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)
        
        self.wavlm_dim = self.wavlm.config.hidden_size
        self.num_wavlm_layers = self.wavlm.config.num_hidden_layers + 1  # +1 pour embedding
        
        # Freeze WavLM
        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.wavlm.eval()
            logger.info("[WavLM2AudioImproved] WavLM frozen")
        
        # Layer weights pour weighted sum (apprenables)
        if use_weighted_layers:
            self.layer_weights = nn.Parameter(
                torch.ones(self.num_wavlm_layers) / self.num_wavlm_layers
            )
            logger.info(f"[WavLM2AudioImproved] Layer weights: {self.num_wavlm_layers} layers")
        
        # Adapter amélioré
        self.adapter = WavLMAdapterImproved(
            wavlm_dim=self.wavlm_dim,
            hidden_dim=hidden_dim,
            num_layers=num_adapter_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            use_snake=use_snake,
        )
        
        # Generator amélioré
        self.generator = HiFiGeneratorImproved(
            hidden_dim=hidden_dim,
            use_snake=use_snake,
        )
        
        # Stats
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[WavLM2AudioImproved] Total params: {total_params:,}")
        logger.info(f"[WavLM2AudioImproved] Trainable params: {trainable_params:,}")
        logger.info("="*80)
    
    def forward(self, audio):
        """
        Args:
            audio: (B, T) waveform [-1, 1]
        Returns:
            (B, T) reconstructed audio
        """
        B, T = audio.shape
        
        # Extraction WavLM
        if self.freeze_wavlm:
            with torch.no_grad():
                outputs = self.wavlm(
                    audio,
                    output_hidden_states=self.use_weighted_layers
                )
        else:
            outputs = self.wavlm(
                audio,
                output_hidden_states=self.use_weighted_layers
            )
        
        # Features: weighted sum ou dernière couche
        if self.use_weighted_layers:
            hidden_states = outputs.hidden_states  # Tuple de (num_layers+1) tenseurs
            weights = F.softmax(self.layer_weights, dim=0)
            features = sum(w * h for w, h in zip(weights, hidden_states))
        else:
            features = outputs.last_hidden_state
        
        # Adapter
        adapted = self.adapter(features)  # (B, hidden_dim, T')
        
        # Generator
        reconstructed = self.generator(adapted)  # (B, 1, T_out)
        reconstructed = reconstructed.squeeze(1)  # (B, T_out)
        
        # Interpolation si nécessaire
        if reconstructed.shape[1] != T:
            reconstructed = F.interpolate(
                reconstructed.unsqueeze(1),
                size=T,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        return reconstructed
    
    def remove_weight_norm(self):
        """Enlève WeightNorm pour l'inférence optimisée."""
        self.adapter.remove_weight_norm()
        self.generator.remove_weight_norm()


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Test du modèle amélioré")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Créer le modèle (utilise HuggingFace hub si pas de fichier local)
    try:
        model = WavLM2AudioImproved(
            wavlm_model_name="microsoft/wavlm-base-plus",
            hidden_dim=256,
            use_weighted_layers=True,
            use_snake=False,
        ).to(device)
    except Exception as e:
        print(f"Erreur de chargement WavLM: {e}")
        print("Skipping test...")
        exit(0)
    
    model.eval()
    
    # Test forward
    batch_size = 2
    seq_len = 16000  # 1 seconde
    
    x = torch.randn(batch_size, seq_len, device=device).clamp(-1, 1)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Vérifications
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert torch.isfinite(y).all(), "Non-finite values in output"
    assert y.min() >= -1.0 and y.max() <= 1.0, "Output out of range"
    
    print("\n✅ Test réussi!")
