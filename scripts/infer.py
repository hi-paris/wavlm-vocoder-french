"""
Inference Script — WavLM2Audio (modèle du papier JEP 2026)
==========================================================

Usage — checkpoint local:
    python scripts/infer.py \
        --checkpoint outputs/checkpoints/checkpoint_step180000_infer.pt \
        --input_file /path/to/audio.wav \
        --output_dir ./generated

Usage — HuggingFace Hub:
    python scripts/infer.py \
        --hf_repo hi-paris/wavlm-vocoder-french \
        --hf_filename checkpoint_step180000_infer.pt \
        --input_file /path/to/audio.wav \
        --output_dir ./generated
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.models_improved import WavLM2AudioImproved
from src.utils.audio import load_audio, save_audio


# Config par défaut correspondant au checkpoint HF
_DEFAULT_CONFIG = {
    "wavlm_model_name": "microsoft/wavlm-base-plus",
    "hidden_dim": 256,
    "num_adapter_layers": 6,
    "kernel_size": 7,
    "freeze_wavlm": True,
    "dropout": 0.1,
    "use_weighted_layers": True,
    "use_snake": False,
}

_DEFAULT_SAMPLE_RATE = 16000
_DEFAULT_CHUNK_SIZE = 320000


def download_hf_checkpoint(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")
    print(f"  Downloading '{filename}' from {repo_id} ...")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"  -> {local_path}")
    return Path(local_path)


def load_model(checkpoint_path: Path, wavlm_path: str, device: torch.device):
    """Charge WavLM2AudioImproved depuis un checkpoint infer."""
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Récupère la config du checkpoint si disponible
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    # Fusionne avec les defaults
    merged = {**_DEFAULT_CONFIG, **model_cfg}

    # Override wavlm_model_name si chemin local fourni
    if wavlm_path:
        merged["wavlm_model_name"] = wavlm_path
    elif not Path(merged["wavlm_model_name"]).exists():
        # Le chemin stocké dans le checkpoint n'existe pas sur cette machine
        # → on garde "microsoft/wavlm-base-plus" pour téléchargement HF
        merged["wavlm_model_name"] = "microsoft/wavlm-base-plus"

    print(f"  WavLM path: {merged['wavlm_model_name']}")

    model = WavLM2AudioImproved(
        wavlm_model_name=merged["wavlm_model_name"],
        hidden_dim=merged["hidden_dim"],
        num_adapter_layers=merged["num_adapter_layers"],
        kernel_size=merged["kernel_size"],
        freeze_wavlm=merged["freeze_wavlm"],
        dropout=merged["dropout"],
        use_weighted_layers=merged["use_weighted_layers"],
        use_snake=merged["use_snake"],
    ).to(device)

    # Charge les poids — clé generator_state_dict (format infer)
    state_dict = ckpt.get("generator_state_dict") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint has no generator_state_dict or model_state_dict. Keys: {list(ckpt.keys())}")

    # Strip DDP prefix si présent
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARNING] Missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  [WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model.eval()
    if hasattr(model, "wavlm"):
        model.wavlm.eval()

    step = ckpt.get("step", "?")
    print(f"  [OK] Model loaded — step={step}")
    return model


def process_audio(model, audio: torch.Tensor, chunk_size: int, device: torch.device) -> torch.Tensor:
    """Inférence avec chunking pour les longs fichiers."""
    model.eval()
    if isinstance(device, str):
        device = torch.device(device)

    if len(audio) <= chunk_size:
        with torch.no_grad():
            out = model(audio.unsqueeze(0).to(device))
        return out.squeeze(0).cpu()

    overlap = chunk_size // 4
    step = chunk_size - overlap
    num_chunks = (len(audio) - overlap + step - 1) // step
    chunks_out = []

    for i in range(num_chunks):
        start = i * step
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        if len(chunk) < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - len(chunk)))
        with torch.no_grad():
            out = model(chunk.unsqueeze(0).to(device)).squeeze(0).cpu()
        actual = end - start
        if i == 0:
            chunks_out.append(out[:chunk_size - overlap // 2])
        elif i == num_chunks - 1:
            chunks_out.append(out[overlap // 2:actual])
        else:
            chunks_out.append(out[overlap // 2:chunk_size - overlap // 2])

    return torch.cat(chunks_out, dim=0)


def main():
    parser = argparse.ArgumentParser(description="WavLM2Audio Inference — JEP 2026")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, help="Chemin local vers le checkpoint (.pt)")
    src.add_argument("--hf_repo", type=str, help="HuggingFace repo id (ex: hi-paris/wavlm-vocoder-french)")

    parser.add_argument("--hf_filename", type=str, default="checkpoint_step180000_infer.pt")
    parser.add_argument("--wavlm_path", type=str, default=None,
                        help="Chemin local vers wavlm-base-plus (optionnel, sinon HF Hub)")

    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input_file", type=str)
    inp.add_argument("--input_dir", type=str)

    parser.add_argument("--output_dir", type=str, default="./generated")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--chunk_size", type=int, default=_DEFAULT_CHUNK_SIZE)
    parser.add_argument("--save_input", action="store_true")

    args = parser.parse_args()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [WARNING] CUDA non disponible — CPU utilisé.")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("  WavLM2Audio — Inference (JEP 2026)")
    print(f"{'='*70}")
    print(f"  Device: {device}")

    # Checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")
    else:
        ckpt_path = download_hf_checkpoint(args.hf_repo, args.hf_filename)

    # Modèle
    model = load_model(ckpt_path, args.wavlm_path, device)

    # Fichiers input
    if args.input_file:
        input_files = [Path(args.input_file)]
    else:
        input_dir = Path(args.input_dir)
        input_files = sorted([
            f for ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a")
            for f in input_dir.glob(f"*{ext}")
        ])
        if args.num_samples:
            input_files = input_files[:args.num_samples]

    if not input_files:
        raise ValueError("Aucun fichier audio trouvé.")

    print(f"  {len(input_files)} fichier(s) à traiter ...\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for input_path in tqdm(input_files, desc="Reconstructing"):
        try:
            audio, sr = load_audio(input_path, target_sr=_DEFAULT_SAMPLE_RATE)
            output = process_audio(model, audio, args.chunk_size, device)

            # Alignement longueur
            tlen = len(audio)
            if len(output) > tlen:
                output = output[:tlen]
            elif len(output) < tlen:
                output = F.pad(output, (0, tlen - len(output)))

            # Normalisation peak
            peak = output.abs().max()
            if peak > 1e-6:
                output = output / peak * 0.95
            output = torch.clamp(output, -1.0, 1.0)

            save_audio(output, output_dir / f"{input_path.stem}_reconstructed.wav", _DEFAULT_SAMPLE_RATE)
            if args.save_input:
                save_audio(audio, output_dir / f"{input_path.stem}_input.wav", _DEFAULT_SAMPLE_RATE)

            n_ok += 1

        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {input_path.name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  [OK] {n_ok}/{len(input_files)} fichiers traités")
    print(f"  [OK] Outputs: {output_dir.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
