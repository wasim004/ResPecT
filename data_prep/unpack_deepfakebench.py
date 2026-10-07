"""Extract the DeepfakeBench test split of DFDC / DFDCP from the preprocessed rgb zips.

Only the frames listed under split 'test' in the official DeepfakeBench JSON are extracted, to
data/faces_dfb/<DS>/test/<real|fake>/<video>/NNN.png (same layout as dfb_extract.py output).
The zip is deleted only if every listed test frame was found and written (user rule: verify, then delete).
"""
import json, os, sys, zipfile

DL = os.environ.get("DOWNLOADS", "downloads")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "faces_dfb")
KEEP_JSON = os.path.join(os.environ.get("DATASETS", "datasets"), "DeepfakeBench")

for ds in sys.argv[1:] or ["DFDC", "DFDCP"]:
    meta = json.load(open(f"{DL}/{ds}.json"))[ds]
    z = zipfile.ZipFile(f"{DL}/{ds}.zip")
    names = set(z.namelist())
    # zip members may or may not carry the leading "<DS>/" that the JSON paths have
    def member(p):
        p = p.replace("\\", "/")
        return p if p in names else p.split("/", 1)[1] if p.split("/", 1)[1] in names else None
    want = missing = written = 0
    vids = {"real": 0, "fake": 0}
    for label, splits in meta.items():
        lab = "real" if label.endswith("_Real") else "fake"
        for vid, v in splits.get("test", {}).items():
            vids[lab] += 1
            for p in v["frames"]:
                want += 1
                m = member(p)
                if m is None:
                    missing += 1; continue
                dst = os.path.join(OUT, ds, "test", lab, vid, os.path.basename(p.replace("\\", "/")))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                data = z.read(m)
                with open(dst, "wb") as fh:
                    fh.write(data)
                written += os.path.getsize(dst) == len(data)
    print(f"{ds}: test videos real {vids['real']} fake {vids['fake']} | frames listed {want}, written {written}, missing {missing}",
          flush=True)
    os.makedirs(KEEP_JSON, exist_ok=True)
    os.replace(f"{DL}/{ds}.json", f"{KEEP_JSON}/{ds}.json")
    if missing == 0 and written == want:
        os.remove(f"{DL}/{ds}.zip"); print(f"{ds}: verified -> zip deleted; json kept in {KEEP_JSON}", flush=True)
    else:
        print(f"{ds}: NOT all frames found -> zip KEPT", flush=True)
