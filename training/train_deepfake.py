"""Deepfake protocol (GUATuning Table 2): train on FF++ c23, test cross-dataset. Reports frame AUC and video AUC."""
import argparse, glob, io, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "models"))   # model code (respect.py) lives in models/
from respect import ReSpecT

CROP = 224       # --crop: training/test crop of the 256 face (252 = 18x18 ViT patches, keeps the face border)
MASK_DIR = ""     # --mask_sup: data/faces_dfb_masks root (FF++ manipulation masks aligned to the crops)
PERTURB = ""      # --perturb (eval only): jpegQ | blurS | noiseS, e.g. jpeg50, blur2, noise10
PF_DIVERSE = 0.0 # --pf_diverse: share of soft colour-mismatch (non-resampled) pseudo-fakes
BENIGN = 0.0     # --benign: prob. of benign rescale/re-encode on real training frames
CUR_SET = ""     # name of the set being evaluated (for per-frame dumps)
DUMP = ""        # --dump_preds: CSV of per-video predictions
TTA = False       # flip test-time averaging (eval only): p = (p(x) + p(mirror(x))) / 2
PREPROC = "crop"   # "crop": 224 crop of the 256 face (ReSpecT); "guatuning": official GUATuning aug + bicubic resize
_GUA_AUG = None


def guatuning_aug():
    """Augmentation actually used by the official GUATuning/DeepfakeBench loader (abstract_dataset.init_data_aug_method)."""
    global _GUA_AUG
    if _GUA_AUG is None:
        import warnings; warnings.filterwarnings("ignore")
        import albumentations as A
        _GUA_AUG = A.Compose([
            A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(p=0.3), A.ImageCompression(quality_lower=40, quality_upper=100, p=0.1),
            A.GaussNoise(p=0.1), A.MotionBlur(p=0.1), A.CLAHE(p=0.1), A.ChannelShuffle(p=0.1), A.Cutout(p=0.1),
            A.RandomGamma(p=0.3), A.GlassBlur(p=0.3)])
    return _GUA_AUG


FACES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "faces")


DF40 = os.path.join(os.environ.get("DATASETS", "datasets"), "DF40/ff_test")


def _pick(vids, n, key):
    """Pilot protocol: a FIXED subset of n videos per label dir (seeded by the dir name, identical for every run)."""
    if not n or len(vids) <= n:
        return vids
    return sorted(random.Random(key).sample(vids, n))


def list_frames(ds, split, max_per_video=None, max_videos=0):
    if ds.startswith("DF40_"):  # DF40 fakes (DeepfakeBench crops) + FF++ test reals from the same pipeline
        reals = [it for it in list_frames("FFpp", "test", max_per_video, max_videos) if it[1] == 0]
        fakes = []
        for vd in _pick(sorted(glob.glob(f"{DF40}/{ds[5:]}/fake/*")), max_videos, f"{ds}/fake"):
            fr = sorted(glob.glob(f"{vd}/*.png"))
            if max_per_video:
                fr = fr[:: max(1, len(fr) // max_per_video)][:max_per_video]
            fakes += [(f, 1, vd) for f in fr]
        return reals + fakes
    items = []
    for lab_dir in sorted(glob.glob(f"{FACES}/{ds}/{split}/*")):
        y = 0 if os.path.basename(lab_dir) == "real" else 1
        for vd in _pick(sorted(glob.glob(f"{lab_dir}/*")), max_videos, f"{ds}/{split}/{os.path.basename(lab_dir)}"):
            fr = sorted(glob.glob(f"{vd}/*.png"))
            if max_per_video:
                fr = fr[:: max(1, len(fr) // max_per_video)][:max_per_video]
            items += [(f, y, vd) for f in fr]
    return items


class Frames(Dataset):
    def __init__(self, items, train, aug="light", with_mask=False):
        self.items, self.train, self.aug, self.with_mask = items, train, aug, with_mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, y, vd = self.items[i]
        im = Image.open(f).convert("RGB")
        M = None
        if isinstance(vd, str) and vd.startswith("PF::"):  # RPFS: self-blended pseudo face forgery (label 1)
            import rpfs
            if self.with_mask:
                im, _, M = rpfs.self_blend(im, return_mask=True, diverse=PF_DIVERSE)
                M = M if MASK_DIR else 4 * M * (1 - M)   # region mask (mask head) or blending-boundary map (U-Net)
            else:
                im = rpfs.self_blend(im, diverse=PF_DIVERSE)[0]
        elif self.train and y == 0 and BENIGN > 0 and random.random() < BENIGN:
            # benign resampling on REAL faces (label stays real): arbitrary-ratio rescale (+ JPEG re-encode) so that
            # re-encoded / rescaled real videos (e.g. YouTube) are not mistaken for forgery resampling
            import rpfs
            im = rpfs.benign_resize(im)
            if random.random() < 0.5:
                buf = io.BytesIO(); im.save(buf, "JPEG", quality=random.randint(60, 95)); im = Image.open(buf).convert("RGB")
        if self.with_mask and M is None and MASK_DIR and y == 1 and not (isinstance(vd, str) and vd.startswith("PF::")):
            mp = f.replace(FACES, MASK_DIR, 1)
            if os.path.exists(mp):
                M = np.asarray(Image.open(mp).convert("L"), dtype=np.float32) / 255.  # real manipulation region in [0,1]
        if self.with_mask and M is None:
            M = np.zeros(im.size[::-1], np.float32) if y == 0 else np.full(im.size[::-1], -1, np.float32)
        flipped = False
        if PERTURB and not self.train:  # robustness evaluation: deterministic test-time degradation (both pipelines)
            if PERTURB.startswith("jpeg"):
                buf = io.BytesIO(); im.save(buf, "JPEG", quality=int(PERTURB[4:])); im = Image.open(buf).convert("RGB")
            elif PERTURB.startswith("blur"):
                im = im.filter(ImageFilter.GaussianBlur(float(PERTURB[4:])))
            elif PERTURB.startswith("noise"):
                import zlib
                rng = np.random.default_rng(zlib.crc32(f.encode()))
                a_ = np.asarray(im, dtype=np.float32) + rng.normal(0, float(PERTURB[5:]), (im.size[1], im.size[0], 3))
                im = Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8))
        if PREPROC == "guatuning":  # official pipeline: (train) albumentations aug -> cv2 bicubic resize to 224
            import cv2
            arr = np.asarray(im)
            if self.train:
                arr = guatuning_aug()(image=arr)["image"]
            arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_CUBIC)
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1), y, vd
        if self.train and self.aug == "gua":  # official GUATuning augmentation, but crop (keeps the pixel lattice)
            im = Image.fromarray(guatuning_aug()(image=np.asarray(im))["image"])
            x0, y0 = random.randint(0, 256 - CROP), random.randint(0, 256 - CROP)
        elif self.train and self.aug == "strong":  # GUATuning/DeepfakeBench recipe (guatuning.yaml data_aug)
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                im = im.rotate(random.uniform(-10, 10), resample=Image.BILINEAR)
            if random.random() < 0.5:
                im = im.filter(ImageFilter.GaussianBlur(random.choice([3, 5, 7]) / 6))
            if random.random() < 0.5:
                im = ImageEnhance.Brightness(im).enhance(1 + random.uniform(-0.1, 0.1))
                im = ImageEnhance.Contrast(im).enhance(1 + random.uniform(-0.1, 0.1))
            if random.random() < 0.5:
                buf = io.BytesIO(); im.save(buf, "JPEG", quality=random.randint(40, 100)); im = Image.open(buf).convert("RGB")
            x0, y0 = random.randint(0, 256 - CROP), random.randint(0, 256 - CROP)
        elif self.train:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT); flipped = True
            if random.random() < 0.1:
                im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
            if random.random() < 0.1:
                buf = io.BytesIO(); im.save(buf, "JPEG", quality=random.randint(60, 100)); im = Image.open(buf).convert("RGB")
            x0, y0 = random.randint(0, 256 - CROP), random.randint(0, 256 - CROP)  # crop, not resize: keeps the pixel lattice
        else:
            x0 = y0 = (256 - CROP) // 2
        im = im.crop((x0, y0, x0 + CROP, y0 + CROP))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if self.with_mask:
            M = M[:, ::-1] if flipped else M
            return x, y, vd, torch.from_numpy(np.ascontiguousarray(M[y0:y0 + CROP, x0:x0 + CROP], dtype=np.float32))[None]
        return x, y, vd


def contrast(z, lab, tau, pos_class=None):
    """ForAda Eq.(3)/(4): for anchor i and positive j (same set), -log exp(s_ij)/(exp(s_ij)+sum_{k in opposite set} exp(s_ik)).
    pos_class: only anchors of this class get positives (sample-wise variant pulls reals only)."""
    z = F.normalize(z, dim=-1); s = z @ z.T / tau
    same = lab[:, None] == lab[None, :]; eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = same & ~eye
    if pos_class is not None:
        pos = pos & (lab[:, None] == pos_class)
    neg = ~same
    if pos.sum() == 0 or neg.sum() == 0:
        return z.sum() * 0
    lse_neg = torch.logsumexp(s.masked_fill(~neg, float("-inf")), 1)          # (n,)
    l = torch.logaddexp(s, lse_neg[:, None]) - s                               # (n, n)
    return l[pos & torch.isfinite(lse_neg)[:, None]].mean()


@torch.no_grad()
def evaluate(model, items, bs, workers):
    model.eval()
    if not items:
        raise RuntimeError("evaluation set is empty (faces not extracted yet?)")
    dl = DataLoader(Frames(items, False), bs, shuffle=False, num_workers=workers, pin_memory=True)
    probs, ys, vids = [], [], []
    for x, y, vd in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xc = x.cuda(non_blocking=True)
            p = torch.sigmoid(model(xc).float())
            if TTA:
                p = 0.5 * (p + torch.sigmoid(model(xc.flip(-1)).float()))
        probs.append(p.cpu()); ys.append(y); vids += list(vd)
    probs, ys = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    fauc = roc_auc_score(ys, probs)
    agg = {}
    for p, y, v in zip(probs, ys, vids):
        agg.setdefault(v, [y, []])[1].append(p)
    vy = np.array([a[0] for a in agg.values()]); vp = np.array([np.mean(a[1]) for a in agg.values()])
    if DUMP and DUMP.endswith(".npz"):  # per-frame dump (seed ensembling): one npz per call, keyed by dataset tag
        np.savez_compressed(DUMP.replace(".npz", f"_{CUR_SET}.npz"), p=probs, y=ys, v=np.array(vids))
    elif DUMP:  # per-video mean probability (error analysis; done on DEV splits)
        with open(DUMP, "a") as f:
            for v, a_ in agg.items():
                f.write(f"{v},{a_[0]},{np.mean(a_[1]):.6f},{np.min(a_[1]):.6f},{np.max(a_[1]):.6f}\n")
    return {"AUC": float(fauc), "Vauc": float(roc_auc_score(vy, vp)), "n_frames": len(ys), "n_videos": len(vy)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="respect_full")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l0", type=float, default=1e-3)
    ap.add_argument("--no_lattice", action="store_true")
    ap.add_argument("--no_dlg", action="store_true")
    ap.add_argument("--train_frames", type=int, default=16)
    ap.add_argument("--test_sets", default="FFpp:test,CDFv2:test,DFD:test,DFDCsample:test")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--faces", default="faces", help="data/<faces> dir: faces (YuNet) | faces_dfb (DeepfakeBench crops)")
    ap.add_argument("--target_density", type=float, default=0.0,
                    help=">0: constrained L0, Lagrangian drives expected open-gate fraction to this value")
    ap.add_argument("--gate_lr", type=float, default=0.0, help="separate lr for gate log-alphas (0 = same as --lr)")
    ap.add_argument("--aug", default="light", choices=["light", "strong", "gua"])
    ap.add_argument("--placement", default="block", choices=["block", "attn"])
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--test_frames", type=int, default=0, help="frames per test video (0 = all extracted)")
    ap.add_argument("--fixed_predictors", action="store_true")
    ap.add_argument("--paps_mode", default="fft", choices=["fft", "pool"])
    ap.add_argument("--eval_only", action="store_true", help="skip training; load experiments/<name>/best.pt")
    ap.add_argument("--model", default="respect", choices=["respect"])
    ap.add_argument("--hyper", action="store_true", help="RASP-X: + alignment/uniformity on the hypersphere + slerp aug")
    ap.add_argument("--no_pixmap", action="store_true"); ap.add_argument("--no_rcln", action="store_true")
    ap.add_argument("--preproc", default="crop", choices=["crop", "guatuning"])
    ap.add_argument("--probe_batches", type=int, default=64, help="GUATuning GSP: T probe steps (paper: 2048 samples / 32 = 64)")
    ap.add_argument("--compact", type=float, default=0.0, help="weight of the compact-real one-class loss (backlog E)")
    ap.add_argument("--subspace", type=int, default=0, help="RASP S2: confine attn-adapter updates to the minor-r subspace")
    ap.add_argument("--rpfs", action="store_true", help="RASP S1: self-blended resampling pseudo-forgeries from FF++ reals")
    ap.add_argument("--swa", type=int, default=0, help="test with the uniform weight average of the last K saved epochs")
    ap.add_argument("--tta", action="store_true", help="flip test-time averaging (eval only)")
    ap.add_argument("--save_epochs", action="store_true", help="keep every epoch's trainable weights (checkpoint-rule study)")
    ap.add_argument("--train_videos", type=int, default=0, help="pilot: fixed subset of N videos per FF++ train label dir")
    ap.add_argument("--val_videos", type=int, default=0, help="pilot: fixed subset of N videos per FF++ val label dir")
    ap.add_argument("--test_videos", type=int, default=0, help="pilot: fixed subset of N videos per test label dir")
    ap.add_argument("--stream_drop", type=float, default=0.0, help="DuoSpecT: prob. of dropping the DINOv2 stream per sample")
    ap.add_argument("--unet", action="store_true", help="DuoSpecT: + BoundaryUNet (blending-boundary map, supervised on self-blends)")
    ap.add_argument("--mamba", action="store_true", help="DuoSpecT: + BiMamba selective scan over the evidence grid")
    ap.add_argument("--caps", action="store_true", help="DuoSpecT: capsule routing head instead of the linear head")
    ap.add_argument("--bnd_w", type=float, default=1.0, help="weight of the boundary-map loss (--unet)")
    ap.add_argument("--caps_w", type=float, default=0.5, help="weight of the capsule margin loss (--caps)")
    ap.add_argument("--rank", type=int, default=32, help="CRLI adapter bottleneck rank (ReSpecT)")
    ap.add_argument("--unfreeze", type=int, default=0, help="fine-tune the last K CLIP blocks (high-capacity option)")
    ap.add_argument("--ft_lr_mult", type=float, default=0.1, help="lr multiplier for the unfrozen CLIP blocks")
    ap.add_argument("--pf_ratio", type=float, default=0.25, help="RPFS: share of pseudo-fakes in the batch mix (real stays .5)")
    ap.add_argument("--ckpt", default="", help="eval: load this checkpoint file from the run dir instead of best.pt")
    ap.add_argument("--cosine", action="store_true", help="per-step cosine LR decay to 0 with 3%% warm-up")
    ap.add_argument("--dump_preds", default="", help="eval: append per-video predictions (video,label,mean,min,max) to this CSV")
    ap.add_argument("--benign", type=float, default=0.0, help="prob. of benign rescale + re-encode on REAL train frames")
    ap.add_argument("--crop", type=int, default=224, help="crop size of the 256 face; 252 keeps the face border (18x18 patches)")
    ap.add_argument("--rp_gate", action="store_true", help="zero-init gate on the r_p injection inside CRLI adapters")
    ap.add_argument("--attn_pool", action="store_true", help="r_p-guided attention pooling of CLIP patch tokens in the head")
    ap.add_argument("--lrpm_ortho", type=float, default=0.0, help="weight of the LRPM predictor orthogonality loss")
    ap.add_argument("--pf_diverse", type=float, default=0.0, help="share of RPFS pseudo-fakes made with Lab colour/blur mismatch, no resampling")
    ap.add_argument("--perturb", default="", help="eval-only test perturbation: jpegQ / blurSIGMA / noiseSTD")
    ap.add_argument("--gate_init", type=float, default=0.9, help="DLG initial open probability (lower = stronger stochastic gating)")
    ap.add_argument("--sem_basis", default="", help="R4: path to semantic_basis.pt (frozen-CLIP PCA on FF++ train)")
    ap.add_argument("--sem_k", type=int, default=0, help="R4: number of semantic directions projected out of the head features")
    ap.add_argument("--mask_sup", action="store_true", help="patch-level forgery-region supervision with real FF++ masks")
    ap.add_argument("--mask_dir", default="faces_dfb_masks", help="mask crop root under data/")
    ap.add_argument("--mask_w", type=float, default=1.0, help="weight of the patch-mask loss")
    ap.add_argument("--clip", default="clip-vit-large-patch14-hf", help="CLIP weights folder under code/weights (e.g. clip-vit-large-patch14-336-hf)")
    ap.add_argument("--patch_con", type=float, default=0.0, help="R15: weight of ForAda-style patch-wise contrastive loss")
    ap.add_argument("--samp_con", type=float, default=0.0, help="R15: weight of ForAda-style sample-wise contrastive loss")
    ap.add_argument("--con_tau", type=float, default=0.1, help="R15: contrastive temperature")
    ap.add_argument("--con_thr", type=float, default=0.1, help="R15: patch is fake if mask coverage > thr (ForAda: 10%%)")
    ap.add_argument("--con_n", type=int, default=256, help="R15: max sampled patches per class per batch")
    ap.add_argument("--side", action="store_true", help="R16: ForAda-style side adapter (ViT-Tiny, query tokens, bias read-out)")
    ap.add_argument("--side_l1", type=float, default=1.0, help="R16: boundary MSE weight")
    ap.add_argument("--side_l2", type=float, default=0.5, help="R16: side patch-contrast weight")
    ap.add_argument("--side_l3", type=float, default=0.2, help="R16: sample-contrast weight on the read-out")
    ap.add_argument("--mdae", action="store_true", help="R17: SRM + YCbCr-phase residual tokens (zero-init) added to r_i")
    ap.add_argument("--sspp", action="store_true", help="R17: seam-sensitive mean+max pooling in the head")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after N steps/epoch")
    a = ap.parse_args()
    global FACES, PREPROC, TTA, DUMP, BENIGN, CROP, PF_DIVERSE, PERTURB, MASK_DIR
    FACES = os.path.join(os.path.dirname(FACES), a.faces)
    PREPROC = a.preproc
    TTA = a.tta
    DUMP = a.dump_preds
    BENIGN = a.benign
    CROP = a.crop
    PF_DIVERSE = a.pf_diverse
    PERTURB = a.perturb
    MASK_DIR = os.path.join(os.path.dirname(FACES), a.mask_dir) if (a.mask_sup or a.patch_con > 0 or a.side) else ""
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", a.name)
    os.makedirs(out, exist_ok=True)

    model = ReSpecT(clip_name=os.path.join(REPO, "weights", a.clip), use_lattice=not a.no_lattice, use_dlg=not a.no_dlg, grad_ckpt=a.grad_ckpt,
                    learn_predictors=not a.fixed_predictors, paps_mode=a.paps_mode, placement=a.placement,
                    subspace=a.subspace, r=a.rank, rp_gate=a.rp_gate, attn_pool=a.attn_pool, gate_init=a.gate_init, mask_head=a.mask_sup, con_proj=(a.patch_con > 0 or a.samp_con > 0), side=a.side, mdae=a.mdae, sspp=a.sspp,
                    side_weights=os.path.join(REPO, "weights", "vit_tiny_patch16_in21k.pt") if a.side else "",
                    sem_basis=a.sem_basis, sem_k=a.sem_k).cuda()
    comp = None
    ft_p = []
    if a.unfreeze > 0:  # high-capacity option: fine-tune the last K CLIP blocks (lower lr) on top of the adapters
        for layer in model.layers[-a.unfreeze:]:
            for n_, p_ in layer.named_parameters():
                if "._ad" not in n_ and not n_.endswith(".P"):
                    p_.requires_grad_(True); ft_p.append(p_)
        seen = {id(p_) for p_ in ft_p}
        ft_p = [p_ for i_, p_ in enumerate(ft_p) if id(p_) in seen and seen.discard(id(p_)) is None]
    params = model.trainable_parameters() + (list(comp.parameters()) if comp is not None else [])
    params = [p_ for p_ in params if all(p_ is not q_ for q_ in ft_p)]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M (+ fine-tuned CLIP {sum(p.numel() for p in ft_p) / 1e6:.2f}M)", flush=True)
    gate_p = [p for n, p in model.named_parameters() if n.startswith("gate.") and p.requires_grad]
    other_p = [p for p in params if all(p is not g for g in gate_p)]
    opt = torch.optim.Adam(([{"params": ft_p, "lr": a.lr * a.ft_lr_mult, "weight_decay": a.wd}] if ft_p else []) + [{"params": other_p, "weight_decay": a.wd},
                            {"params": gate_p, "lr": a.gate_lr or a.lr, "weight_decay": 0.0}], lr=a.lr)
    lag = torch.zeros(2, device="cuda", requires_grad=True)  # Lagrange multipliers (lambda1, lambda2), gradient ascent
    lag_opt = torch.optim.Adam([lag], lr=a.gate_lr or a.lr)

    if a.eval_only:
        return test(model, a, out)
    tr = list_frames("FFpp", "train", a.train_frames, a.train_videos)
    ys = np.array([y for _, y, _ in tr])
    w = np.where(ys == 0, 0.5 / (ys == 0).sum(), 0.5 / (ys == 1).sum())  # class-balanced sampling
    if a.rpfs:  # mix fixed a priori: 50 % real, 25 % true fakes, 25 % pseudo-fakes
        n0 = len(tr)
        tr = tr + [(f, 1, "PF::" + vd) for f, y, vd in tr if y == 0]
        kind = np.array([0 if y == 0 else (2 if vd.startswith("PF::") else 1) for _, y, vd in tr])
        w = np.select([kind == 0, kind == 1, kind == 2], [0.5 / (kind == 0).sum(), (0.5 - a.pf_ratio) / (kind == 1).sum(), a.pf_ratio / (kind == 2).sum()])
        ys = np.array([y for _, y, _ in tr])
        print(f"RPFS: {(kind == 2).sum()} pseudo-fake sources added; sampler mix real .5 / fake {0.5 - a.pf_ratio:.2f} / pseudo {a.pf_ratio:.2f}", flush=True)
    dl = DataLoader(Frames(tr, True, a.aug, with_mask=a.unet or a.mask_sup or a.patch_con > 0 or a.side), a.bs, sampler=WeightedRandomSampler(w, n0 if a.rpfs else len(tr)), num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    val = list_frames("FFpp", "val", 8, a.val_videos)
    print(f"train frames {len(tr)} (real {(ys == 0).sum()}, fake {(ys == 1).sum()}), val frames {len(val)}", flush=True)

    best, log, start = -1, [], 0
    ck = f"{out}/resume.pt"  # written after every epoch so a reboot costs at most one epoch
    if os.path.exists(ck):
        r = torch.load(ck, weights_only=False)
        model.load_state_dict(r["model"], strict=False)
        try:
            opt.load_state_dict(r["opt"])
        except ValueError:  # checkpoint from the single-param-group optimizer (run 1 code)
            print("optimizer state not restored (param-group layout changed); Adam moments restart", flush=True)
        if "lag" in r:
            lag.data.copy_(r["lag"].cuda())
        best, log, start = r["best"], r["log"], r["epoch"] + 1
        random.setstate(r["py_rng"]); np.random.set_state(r["np_rng"]); torch.set_rng_state(r["torch_rng"])
        print(f"resumed from epoch {r['epoch']} (best val AUC {best:.4f})", flush=True)
    base_lrs = [g["lr"] for g in opt.param_groups]
    total_steps = a.epochs * (min(len(dl), a.max_steps) if a.max_steps else len(dl))
    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for step, batch in enumerate(dl):
            if a.cosine:  # per-step cosine decay to 0 with 3 % linear warm-up (stabilises the final weights)
                import math
                gs = ep * (total_steps // a.epochs) + step; wu = max(1, int(0.03 * total_steps))
                f = (gs + 1) / wu if gs < wu else 0.5 * (1 + math.cos(math.pi * (gs - wu) / max(1, total_steps - wu)))
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * f
            x, y = batch[0].cuda(non_blocking=True), batch[1].float().cuda()
            M = batch[3].cuda(non_blocking=True) if len(batch) > 3 else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if comp is not None or a.hyper:
                    logit, feat = model(x, return_feat=True)
                else:
                    logit = model(x)
            logit = logit.float()
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if a.hyper:
                loss = loss + hyper_losses(model, feat.float(), y)
            if a.lrpm_ortho > 0 and getattr(model, "lrpm", None) is not None:
                loss = loss + a.lrpm_ortho * model.lrpm.ortho()
            aux = getattr(model, "_aux", {})
            if M is not None and "bnd" in aux:  # pixel-wise boundary BCE on known targets (reals + self-blends)
                known = (M >= 0).float()
                bl = F.binary_cross_entropy_with_logits(aux["bnd"].float(), M.clamp(min=0), reduction="none")
                loss = loss + a.bnd_w * (bl * known).sum() / known.sum().clamp(min=1)
            if a.side and M is not None and "side_bb" in aux:  # R16 ForAda losses on the SIDE branch
                g = aux["side_bb"].shape[-1]
                Mc = M.clamp(min=0); unk = (M < 0).flatten(1).any(1)                     # images without a mask
                Mb = TF.gaussian_blur(Mc, 9, 2.0); bb = 4 * Mb * (1 - Mb)                # blending boundary (paper: 4M(1-M))
                bbg = F.adaptive_avg_pool2d(bb, g); Bm = (F.adaptive_max_pool2d(bb, g) > 1e-3).float()
                keep = (~unk).float()[:, None, None, None]
                pred = torch.sigmoid(aux["side_bb"].float())
                l1 = (((pred * Bm - bbg) ** 2) * keep).sum() / (keep.sum() * g * g).clamp(min=1)   # masked MSE (Eq. 2)
                loss = loss + a.side_l1 * l1
                cov = F.adaptive_avg_pool2d(Mc, g).flatten(1); lab = (cov > a.con_thr).long().reshape(-1)
                known = (~unk)[:, None].expand(-1, g * g).reshape(-1)
                zf = aux["side_con"].float().reshape(-1, aux["side_con"].shape[-1]); idx = []
                for c in (0, 1):
                    ii = torch.nonzero(known & (lab == c)).squeeze(1)
                    if len(ii): idx.append(ii[torch.randperm(len(ii), device=ii.device)[:a.con_n]])
                if len(idx) == 2:
                    ii = torch.cat(idx); loss = loss + a.side_l2 * contrast(zf[ii], lab[ii], a.con_tau)          # Eq. 3
                loss = loss + a.side_l3 * contrast(aux["side_read"].float(), y.long(), a.con_tau, pos_class=0)  # Eq. 4
            if a.patch_con > 0 and M is not None and "pcon" in aux:  # ForAda-style patch-wise contrast (fake vs real patches)
                g = int(aux["pcon"].shape[1] ** 0.5)
                cov = F.adaptive_avg_pool2d(M.clamp(min=0), g).flatten(1)
                known = (F.adaptive_max_pool2d(-M, g).flatten(1) <= 0)
                lab = (cov > a.con_thr).long()
                zf = aux["pcon"].float().reshape(-1, aux["pcon"].shape[-1]); lab = lab.reshape(-1); known = known.reshape(-1)
                idx = []
                for c in (0, 1):
                    ii = torch.nonzero(known & (lab == c)).squeeze(1)
                    if len(ii): idx.append(ii[torch.randperm(len(ii), device=ii.device)[:a.con_n]])
                if len(idx) == 2:
                    ii = torch.cat(idx); loss = loss + a.patch_con * contrast(zf[ii], lab[ii], a.con_tau)
            if a.samp_con > 0 and "scon" in aux:  # sample-wise contrast: pull reals together, push real vs fake (no fake-fake pull)
                loss = loss + a.samp_con * contrast(aux["scon"].float(), y.long(), a.con_tau, pos_class=0)
            if M is not None and "mask" in aux:  # patch-level forgery-region supervision (coverage of each 14x14 patch)
                g = int(aux["mask"].shape[1] ** 0.5)
                tgt = F.adaptive_avg_pool2d(M.clamp(min=0), g).flatten(1)
                known = (F.adaptive_max_pool2d(-M, g).flatten(1) <= 0).float()  # patch has no "unknown" (-1) pixel
                ml = F.binary_cross_entropy_with_logits(aux["mask"].float(), tgt, reduction="none")
                loss = loss + a.mask_w * (ml * known).sum() / known.sum().clamp(min=1)
            if comp is not None:
                loss = loss + a.compact * comp(feat, y.long())
            if model.use_dlg and a.target_density > 0:
                gap = model.gate.density() - a.target_density
                loss = loss + lag[0] * gap + lag[1] * gap ** 2
            elif model.use_dlg:
                loss = loss + a.l0 * model.l0_penalty() / len(model.layers)
            opt.zero_grad(set_to_none=True); lag_opt.zero_grad(set_to_none=True); loss.backward()
            opt.step()
            if lag.grad is not None:
                lag.grad.neg_(); lag_opt.step()  # ascent on the multipliers
            tot += loss.item()
            if step % 100 == 0:
                dens = f" density {model.gate.density().item():.3f}" if model.use_dlg else ""
                print(f"ep{ep} step{step}/{len(dl)} loss {loss.item():.4f}{dens} {time.time() - t0:.0f}s", flush=True)
            if a.max_steps and step + 1 >= a.max_steps:
                break
        v = evaluate(model, val, 64, a.workers)
        gates = model.gate.sample().detach().cpu().numpy().round(2).tolist() if model.use_dlg else None
        rec = {"epoch": ep, "train_loss": tot / (step + 1), "val": v, "gates": gates}
        log.append(rec); print(json.dumps(rec), flush=True)
        if v["AUC"] > best:
            best = v["AUC"]
            torch.save(trainable_state(model), f"{out}/best.pt")
        if a.save_epochs:
            torch.save(trainable_state(model), f"{out}/epoch_{ep}.pt")
        torch.save({"model": trainable_state(model),
                    "opt": opt.state_dict(), "lag": lag.detach().cpu(), "epoch": ep, "best": best, "log": log, "py_rng": random.getstate(),
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        json.dump(log, open(f"{out}/train_log.json", "w"), indent=1)
    json.dump(log, open(f"{out}/train_log.json", "w"), indent=1)
    test(model, a, out)


def hyper_losses(model, z, y, a_w=0.1, u_w=0.5, t_temp=2.0):
    """Hyperspherical geometry (Wang & Isola 2020) + slerp feature augmentation (LNCLIP-DF style):
    alignment pulls same-class embeddings together, uniformity spreads all embeddings over the sphere, and slerp creates
    extra same-class samples on the geodesic between two same-class embeddings, classified by the same head."""
    zn = F.normalize(z, dim=-1)
    d2 = torch.cdist(zn, zn).pow(2)
    same = (y[:, None] == y[None, :]).float() - torch.eye(len(y), device=y.device)
    align = (d2 * same).sum() / same.sum().clamp(min=1)
    unif = torch.log(torch.exp(-t_temp * d2[~torch.eye(len(y), dtype=torch.bool, device=y.device)]).mean() + 1e-8)
    # slerp between random same-class pairs (per-sphere, since z = [unit CLS ; unit artifact])
    idx = torch.randperm(len(y), device=y.device); keep = y[idx] == y
    lo = 0.0
    if keep.any():
        t = torch.rand(int(keep.sum()), 1, device=y.device)
        parts, k = [], 0
        for dim in (1024, z.shape[1] - 1024):
            a, b = F.normalize(z[keep][:, k:k + dim], dim=-1), F.normalize(z[idx][keep][:, k:k + dim], dim=-1)
            th = torch.acos((a * b).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6))
            parts.append((torch.sin((1 - t) * th) * a + torch.sin(t * th) * b) / torch.sin(th)); k += dim
        zs = torch.cat(parts, -1)
        lo = F.binary_cross_entropy_with_logits(model.head(zs.to(model.head.weight.dtype)).squeeze(-1).float(), y[keep])
    return a_w * align + u_w * unif + lo


def trainable_state(model):
    base = _trainable_state(model)
    extra = {n for n, p in model.named_parameters() if p.requires_grad}  # e.g. unfrozen CLIP blocks (--unfreeze)
    sd = model.state_dict()
    base.update({k: sd[k] for k in extra if k not in base})
    return base


def _trainable_state(model):
    if hasattr(model, "dino"):  # DuoSpecT: drop both frozen backbones (+ the base.layers alias of CLIP's blocks)
        return {k: t for k, t in model.state_dict().items() if not k.startswith(("dino.", "base.clip.", "base.layers."))}
    if hasattr(model, "rcln"):  # RASP-X: trainable RC-LNs live inside clip.* -> keep every trainable param + own buffers
        keep = {n for n, p in model.named_parameters() if p.requires_grad}
        return {k: t for k, t in model.state_dict().items() if k in keep or not k.startswith(("clip.", "layers."))}
    if getattr(model, "det", None) is not None:
        keep = {n for n, p in model.named_parameters() if p.requires_grad}
        return {k: t for k, t in model.state_dict().items() if k in keep}
    return {k: t for k, t in model.state_dict().items() if not k.startswith(("clip.", "layers."))}


def test(model, a, out):
    if a.swa:  # SWA: uniform average of the last K epochs (K fixed a priori), not a best-val checkpoint
        eps = sorted(int(f[6:-3]) for f in os.listdir(out) if f.startswith("epoch_") and f.endswith(".pt"))[-a.swa:]
        sds = [torch.load(f"{out}/epoch_{e}.pt") for e in eps]
        sd = {k: (sum(d[k].float() for d in sds) / len(sds)).to(sds[0][k].dtype) if torch.is_floating_point(sds[0][k])
              else sds[-1][k] for k in sds[0]}
        print(f"SWA over epochs {eps}", flush=True)
    else:
        sd = torch.load(f"{out}/{a.ckpt or 'best.pt'}")  # --ckpt epoch_K.pt: epoch-selection study on dev splits
    model.load_state_dict(sd, strict=False)
    res = {}
    for spec in a.test_sets.split(","):
        ds, split = spec.split(":")
        items = list_frames(ds, split, a.test_frames or None, a.test_videos)
        key = ds if split in ("test", "x") else f"{ds}_{split}"  # e.g. CDFv2_dev (dev splits never overwrite test keys)
        global CUR_SET
        CUR_SET = key
        if items:
            res[key] = evaluate(model, items, 64, a.workers); print(key, res[key], flush=True)
    path = f"{out}/test_results{'_%dv' % a.test_videos if a.test_videos else ''}{'_%df' % a.test_frames if a.test_frames else ''}{'_tta' if a.tta else ''}{'_swa%d' % a.swa if a.swa else ''}{'_' + a.ckpt[:-3] if a.ckpt else ''}{'_' + a.perturb if a.perturb else ''}.json"
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(res); json.dump(prev, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
