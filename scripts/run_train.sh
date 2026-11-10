#!/bin/bash
# Runner script for detached fine-tuning of VoxCPM 0.5B Africa on NVIDIA H200
set -e

cd /mnt/volume_d2wey28/projects/vox-cpm-africa
source .venv/bin/activate
export PYTHONPATH="/mnt/volume_d2wey28/projects/vox-cpm-africa/VoxCPM/src:${PYTHONPATH:-}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting VoxCPM Africa Training in detached mode..."

python ./VoxCPM/scripts/train_voxcpm_finetune.py \
    --config configs/finetune_african_ipa.yaml \
    2>&1 | tee -a train_voxcpm_live.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] VoxCPM training completed!"
