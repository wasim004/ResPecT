"""RPFS -- Resampling Pseudo-Forgery Synthesis (RASP stage S1, 2026-09-30).

Turns REAL training images into pseudo-fakes with random resampling pipelines, so the detector learns *resampling in
general* (the ReSpecT hypothesis: every forgery pipeline resamples) instead of one generator's fingerprint.
Uses only the training set's own real images (no extra data).

  resample_fake(img)  : whole image rendered at low resolution and brought back up with a random operator
                        (nearest / bilinear / bicubic / lanczos / transposed-conv-style zero-insertion + smoothing)
                        -> mimics GAN / U-Net / VAE-decoder up-sampling.                 (ProGAN protocol)
  self_blend(img)     : a resampled, colour-shifted, slightly warped copy of the face blended back through a random
                        feathered elliptical mask -> mimics face-swap warp-and-blend (SBI-style). (deepfake protocol)
Both return (PIL image, operator_id) so an operator-identification head can be added later (stage S3).
"""
import random
import cv2
import numpy as np
from PIL import Image

OPS = ["nearest", "bilinear", "bicubic", "lanczos", "tconv"]
_CV = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}


def _upsample(small, size, op):
    h, w = size
    if op != "tconv":
        return cv2.resize(small, (w, h), interpolation=_CV[op])
    # transposed-conv style: zero-insertion x2 steps + a random small smoothing kernel (leaves checkerboard traces)
    x = small.astype(np.float32)
    while x.shape[0] < h or x.shape[1] < w:
        z = np.zeros((x.shape[0] * 2, x.shape[1] * 2, x.shape[2]), np.float32)
        z[::2, ::2] = x
        tent = np.array([[.25, .5, .25], [.5, 1., .5], [.25, .5, .25]], np.float32)  # bilinear tent = ideal x2 interp.
        k = tent * np.random.uniform(0.97, 1.03, (3, 3)).astype(np.float32)          # imperfect learned kernel
        k = k / k.sum() * 4.0  # gain 4 compensates the inserted zeros -> faint checkerboard, like real generators
        x = cv2.filter2D(z, -1, k)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) if x.shape[:2] != (h, w) else x
    return np.clip(x, 0, 255).astype(np.uint8)


def resample_array(a, op=None, scale=None):
    """a: HxWx3 uint8 -> resampled array, op name."""
    op = op or random.choice(OPS)
    scale = scale or random.choice([0.5, 0.5, 0.75])  # 0.25 dropped: too blocky / trivially detectable (visual check)
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    return _upsample(small, (h, w), op), op


def resample_fake(img):
    a, op = resample_array(np.asarray(img.convert("RGB")))
    return Image.fromarray(a), OPS.index(op)


def _color_jitter(a):
    a = a.astype(np.float32)
    a = a * np.random.uniform(0.9, 1.1, (1, 1, 3)) + np.random.uniform(-10, 10, (1, 1, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def _lab_shift(a):
    """Celeb-DF-style colour mismatch: shift L gain and a/b chroma in Lab space, plus a mild synthesis blur."""
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = lab[..., 0] * np.random.uniform(0.92, 1.08) + np.random.uniform(-6, 6)
    lab[..., 1:] += np.random.uniform(-6, 6, (1, 1, 2))
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    s = np.random.uniform(0.3, 1.2)
    return cv2.GaussianBlur(out, (0, 0), s)


def self_blend(img, return_mask=False, diverse=0.0):
    """SBI-style self-blended pseudo face forgery with a resampled source (return_mask: also the blend mask in [0,1]).
    diverse: probability of a NON-resampled source with a Lab colour/blur mismatch and a wide soft mask (smooth,
    high-quality swaps such as Celeb-DF carry colour/blend cues rather than resampling grids)."""
    tgt = np.asarray(img.convert("RGB"))
    h, w = tgt.shape[:2]
    soft = diverse > 0 and random.random() < diverse
    if soft:
        src, op = _lab_shift(tgt), "bilinear"
    else:
        src, op = resample_array(tgt)
        src = _color_jitter(src)
    # small similarity warp (scale / shift) of the source -> blending-boundary misalignment
    s = random.uniform(0.95, 1.05); tx, ty = random.uniform(-0.03, 0.03) * w, random.uniform(-0.03, 0.03) * h
    M = np.float32([[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]])
    src = cv2.warpAffine(src, M, (w, h), flags=_CV[random.choice(["bilinear", "bicubic"])], borderMode=cv2.BORDER_REFLECT)
    # random feathered ellipse over the (aligned, centred) face region of a DeepfakeBench crop
    mask = np.zeros((h, w), np.float32)
    cx, cy = int(w * random.uniform(0.45, 0.55)), int(h * random.uniform(0.50, 0.60))
    ax, ay = int(w * random.uniform(0.22, 0.34)), int(h * random.uniform(0.28, 0.40))
    cv2.ellipse(mask, (cx, cy), (ax, ay), random.uniform(-15, 15), 0, 360, 1.0, -1)
    k = random.choice([41, 61, 81]) if soft else random.choice([15, 21, 31]); mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask *= random.choice([1.0, 1.0, random.uniform(0.5, 1.0)])  # occasionally a partial-opacity blend
    out = mask[..., None] * src.astype(np.float32) + (1 - mask[..., None]) * tgt.astype(np.float32)
    im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return (im, OPS.index(op), mask) if return_mask else (im, OPS.index(op))


# ---------------------------------------------------------------------------------------------- v5: power-of-two rule
# Insight (S1-ProGAN, 2026-09-30): benchmark REAL images are often resized during dataset curation (e.g. CelebA 178x218
# -> 256, FF++ face crops -> 256). Generators up-sample on a power-of-two lattice (x2 per stage); benign resizing uses
# arbitrary ratios. So: pseudo-fakes = exact x2 up-sampling only; reals get arbitrary-ratio resizing, labelled REAL.

def generator_x2(img):
    """Pseudo-fake: render at half resolution and up-sample exactly x2 with a generator-style operator."""
    a, op = resample_array(np.asarray(img.convert("RGB")), scale=0.5)
    return Image.fromarray(a), OPS.index(op)


def generator_pow2(img):
    """v6 pseudo-fake: exact power-of-two up-sampling, x2 or x4 (Glide 64->256 is x4, LDM decoders are x8 = 3 x2 stages).
    Nearest-neighbour only at x2 (x4 nearest is unrealistically blocky, see visual check)."""
    a = np.asarray(img.convert("RGB"))
    if random.random() < 0.5:
        out, op = resample_array(a, scale=0.5)
    else:
        op = random.choice(["bilinear", "bicubic", "lanczos", "tconv"])
        out, _ = resample_array(a, op=op, scale=0.25)
    return Image.fromarray(out), OPS.index(op)


def benign_resize(img):
    """Real-class augmentation: arbitrary-ratio resizing that avoids the x2 lattice (label stays REAL)."""
    a = np.asarray(img.convert("RGB")); h, w = a.shape[:2]
    if random.random() < 0.5:  # enlarge by a non-power-of-two ratio, then crop back (like curated real datasets)
        s = random.uniform(1.1, 1.85)
        big = cv2.resize(a, (int(round(w * s)), int(round(h * s))), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]))
        y0, x0 = random.randint(0, big.shape[0] - h), random.randint(0, big.shape[1] - w)
        out = big[y0:y0 + h, x0:x0 + w]
    else:                      # shrink by an arbitrary ratio and resize back
        s = random.uniform(0.55, 0.9)
        small = cv2.resize(a, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    return Image.fromarray(out)
