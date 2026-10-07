"""Face-crop extraction for the ReSpecT deepfake protocol (FF++ c23 train; CDF-v2 / DFD / DFDC test).

Per video: N evenly spaced frames -> YuNet largest face -> square box enlarged 1.3x -> 256x256 PNG.
PNG is used deliberately: JPEG re-encoding would add 8x8 block artifacts that interfere with the
resampling cue ReSpecT models.
Writes <out>/<dataset>/<split>/<label>/<video_id>/NNN.png and an index CSV per dataset.
"""
import argparse, csv, json, os, sys
from multiprocessing import Pool
import cv2
import numpy as np

DS = os.environ.get("DATASETS", "datasets")
HERE = os.path.dirname(os.path.abspath(__file__))
YUNET = os.path.join(HERE, "weights", "face_detection_yunet_2023mar.onnx")
_det = None


def detector():
    global _det
    if _det is None:
        cv2.setNumThreads(1)
        _det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.6, 0.3, 5000)
    return _det


def crop_face(frame, scale=1.3, size=256):
    h, w = frame.shape[:2]
    d = detector()
    d.setInputSize((w, h))
    _, faces = d.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])[:4]
    cx, cy, s = x + bw / 2, y + bh / 2, max(bw, bh) * scale
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    x1, y1 = int(round(cx + s / 2)), int(round(cy + s / 2))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def process(job):
    path, out_dir, n = job
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= n:
        return out_dir, len(os.listdir(out_dir))
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return out_dir, 0
    idxs = np.unique(np.linspace(0, total - 1, n).astype(int))
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        face = crop_face(fr)
        if face is not None:
            cv2.imwrite(os.path.join(out_dir, f"{i:05d}.png"), face)
            saved += 1
    cap.release()
    return out_dir, saved


def ffpp_jobs(out, n_train, n_test):
    root = f"{DS}/FF++/c23"
    jobs = []
    for split in ["train", "val", "test"]:
        pairs = json.load(open(f"{DS}/FF++/json_files/{split}.json"))
        n = n_train if split == "train" else n_test
        for a, b in pairs:
            for vid in (a, b):
                jobs.append((f"{root}/original_sequences/youtube/c23/videos/{vid}.mp4",
                             f"{out}/FFpp/{split}/real/{vid}", n))
            for m in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
                for vid in (f"{a}_{b}", f"{b}_{a}"):
                    jobs.append((f"{root}/manipulated_sequences/{m}/c23/videos/{vid}.mp4",
                                 f"{out}/FFpp/{split}/{m}/{vid}", n))
    return jobs


def dfd_jobs(out, n):
    root = f"{DS}/FF++/c23"
    jobs = []
    for sub, lab in [("original_sequences/actors", "real"), ("manipulated_sequences/DeepFakeDetection", "fake")]:
        vdir = f"{root}/{sub}/c23/videos"
        for f in sorted(os.listdir(vdir)):
            jobs.append((f"{vdir}/{f}", f"{out}/DFD/test/{lab}/{f[:-4]}", n))
    return jobs


def cdf_jobs(out, n):
    root = f"{DS}/CelebDF2"
    jobs = []
    for line in open(f"{root}/List_of_testing_videos.txt"):
        if not line.strip():
            continue
        lab, rel = line.split()
        lab = "real" if lab == "1" else "fake"  # CDF list convention: 1 = real, 0 = fake
        vid = rel.replace("/", "__")[:-4]
        jobs.append((f"{root}/{rel}", f"{out}/CDFv2/test/{lab}/{vid}", n))
    return jobs


def dfdc_sample_jobs(out, n):
    root = f"{DS}/DFDC/train_sample_videos"
    meta = json.load(open(f"{root}/metadata.json"))
    return [(f"{root}/{k}", f"{out}/DFDCsample/test/{'fake' if v['label'] == 'FAKE' else 'real'}/{k[:-4]}", n)
            for k, v in sorted(meta.items())]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data", "faces"))
    ap.add_argument("--datasets", default="FFpp,CDFv2,DFD,DFDCsample")
    ap.add_argument("--n_train", type=int, default=16)
    ap.add_argument("--n_test", type=int, default=32)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    for name in a.datasets.split(","):
        jobs = {"FFpp": lambda: ffpp_jobs(a.out, a.n_train, a.n_test),
                "DFD": lambda: dfd_jobs(a.out, a.n_test),
                "CDFv2": lambda: cdf_jobs(a.out, a.n_test),
                "DFDCsample": lambda: dfdc_sample_jobs(a.out, a.n_test)}[name]()
        missing = [j for j in jobs if not os.path.exists(j[0])]
        print(f"[{name}] {len(jobs)} videos, {len(missing)} missing source files", flush=True)
        jobs = [j for j in jobs if os.path.exists(j[0])]
        rows = []
        with Pool(a.workers) as pool:
            for k, (od, cnt) in enumerate(pool.imap_unordered(process, jobs, chunksize=4)):
                rows.append((od, cnt))
                if k % 200 == 0:
                    print(f"[{name}] {k}/{len(jobs)}", flush=True)
        with open(os.path.join(a.out, f"{name}_index.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["video_dir", "n_faces"]); w.writerows(sorted(rows))
        print(f"[{name}] done; {sum(c == 0 for _, c in rows)} videos with no face", flush=True)
