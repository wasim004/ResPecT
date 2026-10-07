#!/bin/bash
# Robustness to test-time perturbations on Celeb-DF v2 (paper: Table 4), for the three trained seeds.
set -e; cd "$(dirname "$0")/.."
B="--faces faces_dfb --train_frames 8 --epochs 3 --placement attn --grad_ckpt --gate_init 0.7 --fixed_predictors --clip clip-vit-large-patch14-336-hf --crop 252 --rpfs"
for P in jpeg50 jpeg30 blur1 blur2 noise5 noise10; do for S in 1024 2025 2026; do
  python training/train_deepfake.py --name respect_s$S $B --seed $S --eval_only --perturb $P --test_sets CDFv2:test
done; done
