#!/bin/bash
# Ablations (paper: Table 5), seed 1024.
set -e; cd "$(dirname "$0")/.."
B="--faces faces_dfb --train_frames 8 --epochs 3 --placement attn --grad_ckpt --gate_init 0.7 --fixed_predictors --clip clip-vit-large-patch14-336-hf --crop 252 --seed 1024"
python training/train_deepfake.py --name abl_noRPFS $B --test_sets CDFv2:dev,DFDC:dev,CDFv2:test,DFD:test,DFDC:test,DFDCP:test,CDFv1:test,UADFV:test,DF40_blendface:x,DF40_e4s:x,DF40_facedancer:x,DF40_fsgan:x,DF40_inswap:x,DF40_mobileswap:x,DF40_simswap:x,DF40_uniface:x
python training/train_deepfake.py --name abl_noLattice $B --rpfs --no_lattice --test_sets CDFv2:dev,DFDC:dev,CDFv2:test,DFD:test,DFDC:test,DFDCP:test,CDFv1:test,UADFV:test,DF40_blendface:x,DF40_e4s:x,DF40_facedancer:x,DF40_fsgan:x,DF40_inswap:x,DF40_mobileswap:x,DF40_simswap:x,DF40_uniface:x
python training/train_deepfake.py --name abl_noDLG $B --rpfs --no_dlg --test_sets CDFv2:dev,DFDC:dev,CDFv2:test,DFD:test,DFDC:test,DFDCP:test,CDFv1:test,UADFV:test,DF40_blendface:x,DF40_e4s:x,DF40_facedancer:x,DF40_fsgan:x,DF40_inswap:x,DF40_mobileswap:x,DF40_simswap:x,DF40_uniface:x
