"""DFDC-dev split (2026-10-01): the Kaggle DFDC train_sample_videos (400 videos: 77 real / 323 fake) are part of the
DFDC TRAINING set, disjoint from the DeepfakeBench DFDC test list used for reporting. Extracted with the identical
DeepfakeBench pipeline (dfb_extract.process, 32 frames, 256 px) to data/faces_dfb/DFDC/dev/{real,fake}/<video>/.
Used only for model selection (epoch check), never for reporting.
"""
import json, os, sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_prep"))
from dfb_extract import process

ROOT = os.path.join(os.environ.get("DATASETS", "datasets"), "DFDC/train_sample_videos")
OUT = os.path.join(os.path.dirname(HERE), "data", "faces_dfb", "DFDC", "dev")
TEST_JSON = os.path.join(os.environ.get("DATASETS", "datasets"), "DeepfakeBench/DFDC.json")

if __name__ == "__main__":
    meta = json.load(open(f"{ROOT}/metadata.json"))
    test = json.load(open(TEST_JSON))["DFDC"]
    test_ids = {k for lab in test.values() for k in lab["test"]}
    ids = {k[:-4] for k in meta}
    assert not ids & test_ids, f"overlap with the DFDC test list: {len(ids & test_ids)}"
    jobs = [(f"{ROOT}/{k}", f"{OUT}/{'fake' if v['label'] == 'FAKE' else 'real'}/{k[:-4]}", 32) for k, v in sorted(meta.items())]
    print(f"DFDC-dev: {len(jobs)} videos, overlap with test list: 0", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        rows = list(pool.imap_unordered(process, jobs, chunksize=2))
    print(f"done; {sum(c == 0 for _, c in rows)} videos with no face", flush=True)
