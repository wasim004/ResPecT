"""Import DeepfakeBench-preprocessed Celeb-DF v1 and UADFV (user downloads, 2026-10-01) into data/faces_dfb.

Both archives contain DeepfakeBench crops (dlib-aligned, 256x256, 32 frames, NNN.png) = same pipeline as faces_dfb.
  CDFv1 : only the official test videos (Celeb-DF/List_of_testing_videos.txt; 1 = real, 0 = fake), as in DeepfakeBench.
  UADFV : all videos are test (DeepfakeBench convention).
Output: data/faces_dfb/<DS>/test/{real,fake}/<video>/NNN.png
"""
import os, shutil, sys, zipfile

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "faces_dfb")
DL = os.environ.get("DOWNLOADS", "downloads")
CDF1_LIST = os.path.join(os.environ.get("DATASETS", "datasets"), "Celeb-DF/List_of_testing_videos.txt")


def copy_dirs(zf, wanted):
    """wanted: {zip_dir_prefix (ending /): out_dir}"""
    n = 0
    for info in zf.infolist():
        for pre, out in wanted.items():
            if info.filename.startswith(pre) and info.filename.endswith(".png"):
                os.makedirs(out, exist_ok=True)
                with zf.open(info) as src, open(os.path.join(out, os.path.basename(info.filename)), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    return n


if __name__ == "__main__":
    # Celeb-DF v1 test videos
    wanted = {}
    for line in open(CDF1_LIST):
        if not line.strip():
            continue
        lab, rel = line.split()
        d, v = rel.split("/"); v = v[:-4]
        wanted[f"{d}/frames/{v}/"] = os.path.join(ROOT, "CDFv1", "test", "real" if lab == "1" else "fake", f"{d}__{v}")
    with zipfile.ZipFile(f"{DL}/Celeb-DF-v1.zip") as zf:
        print("CDFv1 frames copied:", copy_dirs(zf, wanted), "videos wanted:", len(wanted))
    # UADFV: all videos
    with zipfile.ZipFile(f"{DL}/UADFV.zip") as zf:
        vids = sorted({"/".join(n.split("/")[:4]) + "/" for n in zf.namelist() if n.endswith(".png")})
        wanted = {p: os.path.join(ROOT, "UADFV", "test", p.split("/")[1], p.split("/")[3]) for p in vids}
        print("UADFV frames copied:", copy_dirs(zf, wanted), "videos:", len(wanted))
    for ds in ("CDFv1", "UADFV"):
        for lab in ("real", "fake"):
            p = os.path.join(ROOT, ds, "test", lab)
            vs = os.listdir(p) if os.path.isdir(p) else []
            short = [v for v in vs if len(os.listdir(os.path.join(p, v))) < 8]
            print(ds, lab, "videos", len(vs), "with <8 frames:", len(short))
