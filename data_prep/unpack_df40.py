"""Unpack the FF++-domain part of each DF40 test zip, verify it, then delete the zip (user request 2026-09-28).

Output: <DF40>/ff_test/<method>/fake/<video>/<frame>.png  and  <DF40>/ff_test/<method>/landmarks/<video>/...
Handles both layouts in the release: [<method>/]ff/{frames,landmarks}/<vid>/* and the nested mobileswap/ff/frames.zip.
A zip is deleted only if every FF++ member was written and re-read size matches; the cdf-domain part is not kept.
Usage: python unpack_df40.py [method ...]   (default: every *.zip in DF40/Test that is not still downloading)
"""
import glob, io, os, sys, time, zipfile

ROOT = os.path.join(os.environ.get("DATASETS", "datasets"), "DF40")
OUT = os.path.join(ROOT, "ff_test")


def members(z):
    """Yield (zipfile, member name, relative output path) for FF++-domain frames and landmarks."""
    for x in z.namelist():
        p = x.split("/")
        if x.endswith("ff/frames.zip") or x.endswith("ff/landmarks.zip"):  # nested layout (mobileswap)
            kind = "fake" if x.endswith("frames.zip") else "landmarks"
            inner = zipfile.ZipFile(io.BytesIO(z.read(x)))
            for y in inner.namelist():
                if y.endswith("/"):
                    continue
                q = y.split("/")
                for k in ("frames", "landmarks"):
                    if k in q:
                        q = q[q.index(k) + 1:]
                yield inner, y, os.path.join(kind, *q)
        elif "ff" in p and not x.endswith("/"):
            for k, kind in (("frames", "fake"), ("landmarks", "landmarks")):
                if k in p:
                    yield z, x, os.path.join(kind, *p[p.index(k) + 1:])


def stable(path, wait=30):
    """True if the file size is unchanged over `wait` seconds (i.e. not still downloading)."""
    s = os.path.getsize(path); time.sleep(wait); return os.path.getsize(path) == s


for f in sorted(glob.glob(f"{ROOT}/Test/*.zip")):
    m = os.path.basename(f)[:-4]
    if len(sys.argv) > 1 and m not in sys.argv[1:]:
        continue
    if not stable(f):
        print(f"{m}: still growing, skipped", flush=True); continue
    try:
        z = zipfile.ZipFile(f); z.testzip() is None or print(f"{m}: CRC warning", flush=True)
    except zipfile.BadZipFile as e:
        print(f"{m}: bad/incomplete zip ({e}), skipped", flush=True); continue
    n = bad = 0
    for src, name, rel in members(z):
        dst = os.path.join(OUT, m, rel)
        data = src.read(name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(data)
        bad += os.path.getsize(dst) != len(data)
        n += 1
    frames = len(glob.glob(f"{OUT}/{m}/fake/*/*.png"))
    vids = len(glob.glob(f"{OUT}/{m}/fake/*"))
    if n and not bad and frames:
        os.remove(f)
        print(f"{m}: {n} files written ({vids} videos, {frames} frames), verified -> zip deleted", flush=True)
    else:
        print(f"{m}: {n} files, {bad} size mismatches, {frames} frames -> zip KEPT", flush=True)
print("done", flush=True)
