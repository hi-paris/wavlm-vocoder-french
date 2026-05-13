#!/bin/bash
#SBATCH --job-name=test_wavlm_infer
#SBATCH --output=/lustre/fsn1/projects/rech/lsq/umv83if/repos/wavlm-vocoder-french/logs/test_infer_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/lsq/umv83if/repos/wavlm-vocoder-french/logs/test_infer_%j.err
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --account=lsq@a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00

set -e

module purge
module load arch/a100
module load cuda

REPO=/lustre/fsn1/projects/rech/lsq/umv83if/repos/wavlm-vocoder-french
CONDA_ENV=/lustre/fsn1/projects/rech/lsq/umv83if/conda_envs/wavlm_env
WAVLM_PATH=/lustre/fsn1/projects/rech/lsq/umv83if/hf_models/wavlm-base-plus
CHECKPOINT=$REPO/outputs/checkpoints/checkpoint_step180000_infer.pt

export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$REPO:$PYTHONPATH"
export HF_HOME=/lustre/fsn1/projects/rech/lsq/umv83if/.cache/huggingface
export TORCH_HOME=/lustre/fsn1/projects/rech/lsq/umv83if/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p $REPO/logs

cd $REPO

echo "=== Test 1 : import + forward batch_size=1 ==="
python -c "
import torch
from src.models.models_improved import WavLM2AudioImproved

model = WavLM2AudioImproved(
    wavlm_model_name='$WAVLM_PATH',
    hidden_dim=256,
    num_adapter_layers=6,
    freeze_wavlm=True,
    use_weighted_layers=True,
    use_snake=False,
).cuda()
model.eval()
x = torch.randn(1, 16000).cuda()
with torch.no_grad():
    y = model(x)
print(f'[OK] input {x.shape} -> output {y.shape}')
"

echo "=== Test 2 : chargement checkpoint ==="
python -c "
import torch
ckpt = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
print('[OK] Keys:', list(ckpt.keys()))
print('[OK] Step:', ckpt.get('step'))
"

echo "=== Test 3 : inference réelle ==="
python scripts/infer.py \
    --checkpoint $CHECKPOINT \
    --wavlm_path $WAVLM_PATH \
    --input_file $REPO/outputs/samples/segment_ph3.wav \
    --output_dir $REPO/outputs/test_infer_output \
    --device cuda \
    --save_input

echo "=== ALL TESTS PASSED ==="
