# ReSpecT: Resampling-Spectrum-Conditioned Tuning of Vision Foundation Models for Generalizable Face Forgery Detection

Code of the paper *"ReSpecT: Resampling-Spectrum-Conditioned Tuning of Vision Foundation Models for Generalizable Face
Forgery Detection"* (under review).

> **Release status.** This first release contains the **training code** and the **data-preparation and split code**.
> The **model code** and the **trained weights** will be added to `models/` in a later release.

Face-swapping pipelines share one operation, whatever generator synthesizes the face: the new face is warped, resized and
blended into the target frame, that is, it is **resampled**. ReSpecT measures this resampling evidence for every image
patch and uses it to condition lightweight adapters inside a frozen CLIP ViT-L/14-336 encoder. Only 9.49 M parameters
are trained.

## Repository layout

```
training/     train_deepfake.py   training and evaluation (frames, video scores, AUC, test-time perturbations)
              rpfs.py             resampling pseudo-forgery synthesis (RPFS), used during training
data_prep/    face extraction with the DeepfakeBench pipeline and import of the released benchmark crops
splits/       development splits used for all design decisions (scripts + video lists)
scripts/      commands for the final model, the ablations and the robustness tests
models/       model code and trained weights (to be released)
```

## Installation

```bash
conda create -n respect python=3.10 -y && conda activate respect
pip install -r requirements.txt
```

## Data

We use the face crops of the [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) pipeline: dlib detection,
81-point landmarks, 5-point alignment, a 1.3x face box, 256x256 crops and 32 uniformly sampled frames per video.

1. Obtain the datasets from their providers under their own licences: FaceForensics++ (c23), Celeb-DF v1/v2,
   DeepFakeDetection, DFDC, DFDC Preview, UADFV and [DF40](https://github.com/YZY-stack/DF40). The datasets are not
   redistributed here.
2. Copy DeepfakeBench's `preprocessing/preprocess.py` and the dlib model `shape_predictor_81_face_landmarks.dat` into
   `data_prep/deepfakebench_prep/`. Neither is redistributed here.
3. Set `DATASETS` to the folder with the raw datasets and `DOWNLOADS` to the folder with the DeepfakeBench / DF40
   release archives, then run:
   ```bash
   export DATASETS=/path/to/datasets DOWNLOADS=/path/to/archives
   python data_prep/dfb_extract.py              # face crops -> data/faces_dfb/<dataset>/<split>/<label>/<video>/NNN.png
   python data_prep/unpack_deepfakebench.py     # released crops (DFDC, DFDCP, ...)
   python data_prep/unpack_df40.py              # DF40 face-swap subset built on FF++ identities
   python data_prep/import_dfb_extra.py         # Celeb-DF v1, UADFV
   ```

## Development splits

All design decisions were taken on two development sets that share no video with any test list:
- **CDF-dev:** 299 Celeb-DF v2 videos outside its official test list (`splits/CDFv2_dev_videos.csv`, created by
  `splits/make_cdf_dev.py`);
- **DFDC-dev:** the 381 videos of the public DFDC training sample (`splits/DFDC_dev_videos.csv`, created by
  `splits/make_dfdc_dev.py`).

```bash
python splits/make_cdf_dev.py
python splits/make_dfdc_dev.py
```

## Training and evaluation

Training requires the model code in `models/` (to be released).

```bash
bash scripts/train_eval_final.sh   # final model, seeds 1024 / 2025 / 2026
bash scripts/ablations.sh          # ablations (seed 1024)
bash scripts/robustness.sh         # JPEG, blur and noise on Celeb-DF v2
```

The final configuration is:

```bash
python training/train_deepfake.py --name respect_s1024 --seed 1024 --faces faces_dfb --train_frames 8 --epochs 3 \
  --placement attn --grad_ckpt --gate_init 0.7 --fixed_predictors --rpfs \
  --clip clip-vit-large-patch14-336-hf --crop 252 --test_sets CDFv2:dev,DFDC:dev
```

Training details:
- the training mix is, on average, 50% real FF++ frames, 25% FF++ forgeries and 25% RPFS pseudo-forgeries;
- Adam with a constant learning rate of 2e-4, batch size 32, 3 epochs, 8 frames per training video;
- the checkpoint is chosen by the best frame AUC on the FF++ validation split;
- a video is scored by the mean of its frame logits.

The CLIP ViT-L/14-336 weights are expected in `weights/clip-vit-large-patch14-336-hf` (Hugging Face format).

## Citation

The paper is under review; the citation will be added here.

## Licence and acknowledgements

The code is released under the MIT licence (see `LICENSE`).

We build on:
- [CLIP](https://github.com/openai/CLIP);
- the [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) preprocessing and test lists;
- the [DF40](https://github.com/YZY-stack/DF40) benchmark.

Their licences apply to their components and to the datasets.
