"""CDF-dev split (2026-10-01): a Celeb-DF v2 DEVELOPMENT set for model selection in Celeb-DF-targeted pilots, so the
official CDF test list (List_of_testing_videos.txt, 518 videos) is never used to choose a configuration.

Videos are disjoint from the test list. Identity-disjoint videos (no idNN shared with any test video) are taken first;
the remainder are random non-test videos (fixed seed). 150 real (Celeb-real + YouTube-real) + 150 fake (Celeb-synthesis).
Extraction is identical to the test set (DeepfakeBench dlib pipeline, 32 frames, 256 px) via dfb_extract.process.
Output: data/faces_dfb/CDFv2/dev/{real,fake}/<video>/NNN.png  (+ CDFv2_dev_index.csv)
"""
import csv, os, random, re, sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_prep"))
from dfb_extract import process

ROOT = os.path.join(os.environ.get("DATASETS", "datasets"), "CelebDF2")
OUT = os.path.join(os.path.dirname(HERE), "data", "faces_dfb", "CDFv2", "dev")


def ids(path):
    return set(re.findall(r"id\d+", os.path.basename(path)))


def pick(cands, n, test_ids, rng):
    disj = sorted(v for v in cands if not (ids(v) & test_ids))
    rest = sorted(v for v in cands if ids(v) & test_ids)
    out = disj[:n]
    return out + rng.sample(rest, n - len(out)) if len(out) < n else out


if __name__ == "__main__":
    rng = random.Random("cdf_dev_2026")
    test = {l.split()[1] for l in open(f"{ROOT}/List_of_testing_videos.txt") if l.strip()}
    tid = set().union(*[ids(t) for t in test])
    nontest = lambda d: [f"{d}/{v}" for v in os.listdir(f"{ROOT}/{d}") if v.endswith(".mp4") and f"{d}/{v}" not in test]
    real = pick(nontest("Celeb-real"), 75, tid, rng) + sorted(rng.sample(sorted(nontest("YouTube-real")), 75))
    fake = pick(nontest("Celeb-synthesis"), 150, tid, rng)
    assert not (set(real) | set(fake)) & test, "dev split overlaps the test list"
    jobs = [(f"{ROOT}/{v}", f"{OUT}/{lab}/{v.replace('/', '__')[:-4]}", 32) for lab, vs in (("real", real), ("fake", fake)) for v in vs]
    print(f"CDF-dev: {len(real)} real, {len(fake)} fake; id-disjoint real {sum(not (ids(v) & tid) for v in real)}, "
          f"fake {sum(not (ids(v) & tid) for v in fake)}", flush=True)
    rows = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for k, r in enumerate(pool.imap_unordered(process, jobs, chunksize=2)):
            rows.append(r)
            if k % 50 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    with open(os.path.join(os.path.dirname(OUT), "..", "CDFv2_dev_index.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_dir", "n_faces"]); w.writerows(sorted(rows))
    print(f"done; {sum(c == 0 for _, c in rows)} videos with no face", flush=True)
