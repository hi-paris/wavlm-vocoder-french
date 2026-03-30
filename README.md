# WavLM Vocoder for French 🎙️🇫🇷

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

> **WavLM-to-Audio Vocoding in French: Layer Ablation Study and Adversarial Supervision for Continuous Voice Conversion**  
> Neural vocoder for reconstructing high-quality French speech from WavLM representations

> **News — March 2026:** This work was **accepted at JEP 2026**.

🎯 **Goal**: Stage 1 foundation for continuous voice conversion in WavLM latent space

🔗 **Demo**: [WavLM2Audio Demo](https://hi-paris.github.io/wavlm2audio-demo/)  
🤗 **Model Card**: [Hugging Face](https://huggingface.co/hi-paris/wavlm-vocoder-french)

---
## 🎯 Overview

This repository implements a neural vocoder that reconstructs waveform audio from frozen **WavLM-Base+** representations, specifically trained and evaluated on French speech corpora.

It accompanies our **JEP 2026 accepted paper** and serves as a **stage-1 reconstructive decoder** for future continuous voice conversion in WavLM latent space.
---

### Key Features

- ✅ **WavLM-Base+ Integration**: Frozen 12-layer transformer encoder (768-dim)
- ✅ **HiFi-GAN Generator**: Progressive upsampling (×320) with multi-receptive field residual blocks
- ✅ **Layer Ablation Study**: Systematic evaluation of N last layers (N=1...12)
- ✅ **Learned Layer Fusion**: Weighted combination vs. simple averaging
- ✅ **Adversarial Training**: MPD/MSD discriminators + Feature Matching
- ✅ **French Corpora**: SIWIS (10.9h) + M-AILABS (160.7h) + Common Voice (66.7h) = 238.3h

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│  Audio (16kHz) → WavLM-Base+ (frozen) → Layer Selection        │
│       ↓                                                          │
│  Learned Fusion (α₁h₁ + ... + αₙhₙ) → Adapter (768→256)       │
│       ↓                                                          │
│  HiFi-GAN Generator (×320 upsampling) → Reconstructed Audio    │
│       ↓                                                          │
│  [Optional] MPD/MSD Discriminators + Feature Matching          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📄 Associated paper

**WavLM-to-Audio Vocoding in French: Layer Ablation Study and Adversarial Supervision for Continuous Voice Conversion**  
*Nassima Ould Ouali, Awais Hussain Sani, Reda Dehak, Eric Moulines*  
**Accepted at JEP 2026**

This repository contains the codebase associated with the accepted paper, including training, evaluation, ablation, and inference utilities.

---

## 📊 Results Summary

| Configuration | MCD↓ | Mel-L1↓ | PESQ↑ | STOI↑ | V/UV F1↑ | F0 RMSE↓ | F0 Corr↑ |
|--------------|------|---------|-------|-------|----------|----------|----------|
| **No GAN** | 9.72 | 1.55 | 1.11 | 0.74 | 0.878 | 10.1 | 0.83 |
| **+MPD/MSD+FM** | **8.43** | **1.17** | **1.28** | **0.86** | **0.932** | **7.7** | **0.96** |
| **Gain** | -13.3% | -24.5% | +15.3% | +16.2% | +6.1% | -23.8% | +15.7% |

> **Key Findings**:
> - Adversarial supervision (GAN) provides **consistent gains** across all metrics
> - Layers 7-12 capture most phonetic-prosodic information
> - Learned layer fusion outperforms fixed single-layer extraction

Full ablation results: [`results_ablation_N1to6.csv`](results_ablation_N1to6.csv) and [`results_FINAL.csv`](results_FINAL.csv)

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/hi-paris/wavlm-vocoder-french.git
cd wavlm-vocoder-french
pip install -e .
```

For evaluation metrics (PESQ, STOI, F0):
```bash
pip install -e ".[eval]"
```

### 2. Training
```bash
# Single GPU — no GAN baseline
python scripts/train.py --config configs/experiments/no_gan.yaml

# Single GPU — full GAN model
python scripts/train.py --config configs/experiments/gan.yaml

# Multi-GPU with torchrun
torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/experiments/gan.yaml
```

### 3. Layer Ablation
```bash
python scripts/run_ablation.py \
    --base_config configs/experiments/ablation_layers.yaml \
    --output_dir outputs/ablation \
    --layers 1,2,3,4,6,9,12
```

### 4. Inference
```bash
python scripts/infer.py \
    --checkpoint outputs/checkpoints/checkpoint_best.pt \
    --input_dir /path/to/audio \
    --output_dir outputs/samples \
    --num_samples 10
```

### 5. Evaluation
```bash
python scripts/eval.py \
    --checkpoint outputs/checkpoints/checkpoint_best.pt \
    --test_dir /path/to/test/audio \
    --output_dir outputs/eval_results
```

---

## 📁 Repository Structure
```
wavlm-vocoder-french/
├── src/
│   ├── models/
│   │   ├── adapter.py          # WavLM adapter (768→256) + LayerFusion
│   │   ├── generator.py        # HiFi-GAN generator (×320 upsampling)
│   │   ├── discriminator.py    # MPD/MSD discriminators
│   │   └── wavlm_vocoder.py    # Main vocoder (WavLM + adapter + generator)
│   ├── losses/
│   │   ├── reconstruction.py   # L1 + Multi-Scale STFT losses
│   │   ├── gan.py              # Adversarial + Feature Matching losses
│   │   └── combined.py         # Combined loss (reconstruction + GAN)
│   ├── data/
│   │   ├── dataset.py          # AudioDataset (segmentation, normalization)
│   │   └── collate.py          # Collate function for DataLoader
│   ├── trainers/
│   │   └── trainer.py          # DDP/AMP trainer with checkpointing
│   └── utils/
│       ├── audio.py            # load/save audio, chunked inference
│       ├── audio_processing.py # Audio processing utilities
│       ├── checkpoint.py       # Save/load checkpoints
│       ├── config.py           # YAML config loading with inheritance
│       └── logging.py          # Logging setup
├── configs/
│   ├── base.yaml               # Base hyperparameters
│   └── experiments/
│       ├── no_gan.yaml         # Baseline (spectral losses only)
│       ├── gan.yaml            # Full model (MPD/MSD + FM)
│       └── ablation_layers.yaml # Layer sweep experiments
├── scripts/
│   ├── train.py                # Training entry point
│   ├── infer.py                # Inference on audio files
│   ├── eval.py                 # Evaluation script
│   ├── run_ablation.py         # Layer ablation study runner
│   └── analyze_ablation_results.py # Ablation results analysis
├── tests/
│   ├── test_models.py          # Model architecture tests
│   ├── test_losses.py          # Loss function tests
│   ├── test_dataset.py         # Dataset/collate tests
│   └── test_training.py        # Training components tests
├── paper_assets/
│   └── docs/
│       ├── figures/            # Ablation plots (PDF/PNG)
│       └── layer_importance_table.tex
├── outputs/
│   ├── samples/sweep_outputs/  # Audio samples at ckpt 160k/180k/200k
│   └── logs/                   # TensorBoard event files
├── results_ablation_N1to6.csv  # Ablation study results
├── results_FINAL.csv           # Final model results
├── pyproject.toml              # Package config + black/ruff/pytest settings
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── LICENSE                     # MIT License
└── CITATION.bib                # BibTeX citation
```

---

## 🔬 Key Experiments

### GAN vs. No-GAN
```bash
python scripts/train.py --config configs/experiments/no_gan.yaml
python scripts/train.py --config configs/experiments/gan.yaml
```
**Result**: GAN provides consistent impovements across spectral, intelligibility and prosodic metrics .

### Layer Ablation (N=1..12)
```bash
python scripts/run_ablation.py \
    --base_config configs/experiments/ablation_layers.yaml \
    --layers 1,2,3,4,6,9,12
```
**Result**: N=9 layers (7-12) is optimal.

### Analyze Ablation Results
```bash
python scripts/analyze_ablation_results.py \
    --output_dir outputs/ablation
```

---

## 📦 Pretrained Checkpoints

| Model | Layers | GAN | MCD | PESQ |
|-------|--------|-----|-----|------|
| Baseline | 12 | ❌ | 9.72 | 1.11 |
| **Best (N=9)** | 9 | ✅ | **8.43** | **1.28** |
| Lightweight (N=6) | 6 | ✅ | 8.89 | 1.21 |

> Checkpoints not included in this repository due to size (~1.4GB each).
> See [`outputs/samples/sweep_outputs/`](outputs/samples/sweep_outputs/) for audio samples.

---

## 🎓 Citation
```bibtex
@misc{wavlm_vocoder_french_2026,
  title={WavLM-to-Audio Vocoding in French: Layer Ablation Study and Adversarial Supervision for Continuous Voice Conversion},
  author={Nassima Ould Ouali and Awais Hussain Sani and Reda Dehak and Eric Moulines},
  year={2026},
  note={Accepted at JEP 2026}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

- **WavLM**: Microsoft Research ([Chen et al., 2022](https://arxiv.org/abs/2110.13900))
- **HiFi-GAN**: [Kong et al., 2020](https://arxiv.org/abs/2010.05646)
- **Datasets**: SIWIS, M-AILABS, Common Voice

---

## 📧 Contact

For questions, open an issue or contact: **nassima.ould-ouali@ip-paris.fr**

---

## 🗺️ Roadmap

- [x] Stage 1: Reconstruction vocoder (this work)
- [ ] Stage 2: Voice conversion in WavLM latent space
- [ ] Stage 3: Diffusion/Flow-based manipulation
