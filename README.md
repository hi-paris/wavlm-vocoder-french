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

## 🚀 Try it now

```bash
git clone https://github.com/hi-paris/wavlm-vocoder-french.git
cd wavlm-vocoder-french
pip install -e .

# Checkpoint downloads automatically from HF Hub
python scripts/infer.py \
  --hf_repo hi-paris/wavlm-vocoder-french \
  --hf_filename checkpoint_step180000_infer.pt \
  --input_file /path/to/your/audio.wav \
  --output_dir ./generated \
  --device cuda
```

> **Jean Zay / IDRIS users**: compute nodes have no internet access. Download the checkpoint from the login node first:
> ```bash
> python -c "
> from huggingface_hub import hf_hub_download
> hf_hub_download(repo_id='hi-paris/wavlm-vocoder-french',
>                 filename='checkpoint_step180000_infer.pt',
>                 local_dir='./outputs/checkpoints')
> "
> # Then use --checkpoint instead of --hf_repo in your SLURM script
> python scripts/infer.py \
>   --checkpoint outputs/checkpoints/checkpoint_step180000_infer.pt \
>   --input_file /path/to/audio.wav \
>   --output_dir ./generated \
>   --device cuda
> ```

---

## 🎯 Overview

This repository implements a neural vocoder that reconstructs waveform audio from frozen **WavLM-Base+** representations, specifically trained and evaluated on French speech corpora.

It accompanies our **JEP 2026 accepted paper** and serves as a **stage-1 reconstructive decoder** for future continuous voice conversion in WavLM latent space.

### Key Features

- ✅ **WavLM-Base+ Integration**: Frozen 12-layer transformer encoder (768-dim)
- ✅ **HiFi-GAN Generator**: Progressive upsampling (×320) with WeightNorm
- ✅ **Learned Layer Fusion**: Weighted combination of all WavLM layers
- ✅ **Adversarial Training**: MPD/MSD discriminators + Feature Matching
- ✅ **French Corpora**: SIWIS (10.9h) + M-AILABS (160.7h) + Common Voice (66.7h) = 238.3h

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Audio (16kHz) → WavLM-Base+ (frozen) → Learned Layer Fusion   │
│       ↓                                                          │
│  Weighted Sum (α₁h₁ + ... + α₁₃h₁₃) → Adapter (768→256)       │
│       ↓                                                          │
│  HiFi-GAN Generator (×320 upsampling) → Reconstructed Audio    │
│       ↓                                                          │
│  MPD/MSD Discriminators + Feature Matching (training only)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📄 Associated paper

**WavLM-to-Audio Vocoding in French: Layer Ablation Study and Adversarial Supervision for Continuous Voice Conversion**  
*Nassima Ould Ouali, Awais Hussain Sani, Reda Dehak, Eric Moulines*  
**Accepted at JEP 2026**

---

## 📊 Results Summary

| Configuration | MCD↓ | Mel-L1↓ | PESQ↑ | STOI↑ | V/UV F1↑ | F0 RMSE↓ | F0 Corr↑ |
|--------------|------|---------|-------|-------|----------|----------|----------|
| **No GAN** | 9.72 | 1.55 | 1.11 | 0.74 | 0.878 | 10.1 | 0.83 |
| **+MPD/MSD+FM** | **8.43** | **1.17** | **1.28** | **0.86** | **0.932** | **7.7** | **0.96** |
| **Gain** | -13.3% | -24.5% | +15.3% | +16.2% | +6.1% | -23.8% | +15.7% |

Full ablation results: [`results_ablation_N1to6.csv`](results_ablation_N1to6.csv) and [`results_FINAL.csv`](results_FINAL.csv)

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/hi-paris/wavlm-vocoder-french.git
cd wavlm-vocoder-french
pip install -e .
```

### 2. Inference

```bash
# Option 1 — Automatic download from HF Hub
python scripts/infer.py \
  --hf_repo hi-paris/wavlm-vocoder-french \
  --hf_filename checkpoint_step180000_infer.pt \
  --input_file /path/to/audio.wav \
  --output_dir ./generated \
  --device cuda

# Option 2 — Local checkpoint
python scripts/infer.py \
  --checkpoint /path/to/checkpoint_step180000_infer.pt \
  --input_file /path/to/audio.wav \
  --output_dir ./generated \
  --device cuda

# Process a full directory
python scripts/infer.py \
  --hf_repo hi-paris/wavlm-vocoder-french \
  --hf_filename checkpoint_step180000_infer.pt \
  --input_dir /path/to/audio_dir \
  --output_dir ./generated \
  --device cuda
```

### 3. Evaluation

```bash
python scripts/eval.py \
    --checkpoint outputs/checkpoints/checkpoint_step180000_infer.pt \
    --test_dir /path/to/test/audio \
    --output_dir outputs/eval_results
```

---

## 📁 Repository Structure

```
wavlm-vocoder-french/
├── src/
│   ├── models/
│   │   ├── models_improved.py  # WavLM2AudioImproved (full model)
│   │   └── discriminators.py   # MPD/MSD discriminators
│   ├── data/
│   │   ├── dataset.py          # AudioDataset
│   │   └── collate.py          # Collate function
│   ├── trainers/
│   │   └── trainer.py          # DDP/AMP trainer
│   └── utils/
│       ├── audio.py            # load/save audio, chunked inference
│       ├── checkpoint.py       # Save/load checkpoints
│       ├── config.py           # YAML config loading
│       └── logging.py          # Logging setup
├── configs/
│   └── base.yaml               # Base hyperparameters
├── scripts/
│   ├── infer.py                # Inference script
│   ├── train.py                # Training entry point
│   ├── eval.py                 # Evaluation script
│   └── run_ablation.py         # Layer ablation runner
├── tests/
├── results_ablation_N1to6.csv
├── results_FINAL.csv
├── pyproject.toml
├── setup.py
├── LICENSE
└── CITATION.bib
```

---

## 📦 Pretrained Checkpoint

| Model | GAN | MCD↓ | PESQ↑ | Size |
|-------|-----|------|-------|------|
| **Best (step 180000)** | ✅ MPD/MSD+FM | **8.43** | **1.28** | 427 MB |

Download: [hi-paris/wavlm-vocoder-french](https://huggingface.co/hi-paris/wavlm-vocoder-french)

---

## 🎓 Citation

```bibtex
@misc{ouldouali2026wavlm2audiofr,
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
