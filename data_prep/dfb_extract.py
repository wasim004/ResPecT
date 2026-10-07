"""DeepfakeBench-parity face extraction (protocol parity with GUATuning / Effort / DF40).

Uses DeepfakeBench's own `extract_aligned_face_dlib` (deepfakebench_prep/preprocess.py, unmodified):
dlib HOG detector + 81-point landmarks, 5-point similarity alignment, scale 1.3, 256x256.
Frame selection copies DeepfakeBench `facecrop`: np.linspace(0, n-1, 32) over a sequential read,
saved as <frame_idx:03d>.png -- the same naming as the DF40 release.
Output: data/faces_dfb/<DS>/<split>/<label>/<video>/NNN.png   (label: real | <manipulation> | fake)
"""
import argparse, csv, importlib.util, os, sys
from multiprocessing import Pool
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, "deepfakebench_prep")
spec = importlib.util.spec_from_file_location("dfb_preprocess", os.path.join(PREP, "preprocess.py"))
dfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(dfb)
sys.path.insert(0, HERE)
import extract_faces as E  # reuse the dataset/split job lists (same videos as the YuNet extraction)

_det = _pred = None


def models():
    global _det, _pred
    if _det is None:
        import dlib
        cv2.setNumThreads(1)
        _det = dlib.get_frontal_face_detector()
        _pred = dlib.shape_predictor(os.path.join(PREP, "shape_predictor_81_face_landmarks.dat"))
    return _det, _pred


def process(job):
    path, out_dir, n = job
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= n - 2:  # DFB drops frames with no face; allow a few
        return out_dir, len(os.listdir(out_dir))
    det, pred = models()
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return out_dir, 0
    want = set(np.linspace(0, total - 1, n, endpoint=True, dtype=int).tolist())
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for i in range(total):
        if not cap.grab():
            break
        if i not in want:
            continue
        ok, fr = cap.retrieve()
        if not ok:
            continue
        face, lm, _ = dfb.extract_aligned_face_dlib(det, pred, fr)
        if face is not None and lm is not None:
            cv2.imwrite(os.path.join(out_dir, f"{i:03d}.png"), face)
            saved += 1
        if i >= max(want):
            break
    cap.release()
    return out_dir, saved


def remap(jobs, out):
    """Point E's job list (built for data/faces) at data/faces_dfb and use 32 frames everywhere."""
    src = os.path.join(os.path.dirname(HERE), "data", "faces")
    return [(p, o.replace(src, out, 1), 32) for p, o, _ in jobs]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data", "faces_dfb"))
    ap.add_argument("--datasets", default="FFpp,CDFv2,DFD")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default="", help="debug: comma list of substrings; keep jobs whose video path matches")
    a = ap.parse_args()
    base = os.path.join(os.path.dirname(HERE), "data", "faces")
    for name in a.datasets.split(","):
        jobs = {"FFpp": lambda: E.ffpp_jobs(base, 32, 32), "DFD": lambda: E.dfd_jobs(base, 32),
                "CDFv2": lambda: E.cdf_jobs(base, 32), "DFDCsample": lambda: E.dfdc_sample_jobs(base, 32)}[name]()
        jobs = [j for j in remap(jobs, a.out) if os.path.exists(j[0])]
        if a.only:
            jobs = [j for j in jobs if any(s in j[0] for s in a.only.split(","))]
        print(f"[{name}] {len(jobs)} videos", flush=True)
        rows = []
        with Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(process, jobs, chunksize=2)):
                rows.append(r)
                if k % 200 == 0:
                    print(f"[{name}] {k}/{len(jobs)}", flush=True)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"{name}_index.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["video_dir", "n_faces"]); w.writerows(sorted(rows))
        print(f"[{name}] done; {sum(c == 0 for _, c in rows)} videos with no face", flush=True)
