#!/bin/bash
# Final ReSpecT model (paper: Tables 1-3): three training seeds, development sets for selection, then all test sets.
set -e; cd "$(dirname "$0")/.."
B="--faces faces_dfb --train_frames 8 --epochs 3 --placement attn --grad_ckpt --gate_init 0.7 --fixed_predictors --clip clip-vit-large-patch14-336-hf --crop 252 --rpfs"
for S in 1024 2025 2026; do
  python training/train_deepfake.py --name respect_s$S $B --seed $S --test_sets CDFv2:dev,DFDC:dev
  python training/train_deepfake.py --name respect_s$S $B --seed $S --eval_only --test_sets CDFv2:test,DFD:test,DFDC:test,DFDCP:test,CDFv1:test,UADFV:test,DF40_blendface:x,DF40_e4s:x,DF40_facedancer:x,DF40_fsgan:x,DF40_inswap:x,DF40_mobileswap:x,DF40_simswap:x,DF40_uniface:x
done
