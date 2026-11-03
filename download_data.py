"""Download and extract COCO val2017 and the KITTI 2D object benchmark.

Usage:
    python download_data.py coco          # ~1 GB  (val2017 images + annotations)
    python download_data.py kitti         # ~12 GB (left color images + labels)
    python download_data.py coco kitti --root data

KITTI is distributed under CC BY-NC-SA 3.0; register / accept the terms at
https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d before downloading.
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

FILES = {
    "coco": [
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    ],
    "kitti": [
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    ],
}


def _progress(block, block_size, total):
    done = block * block_size
    if total > 0:
        pct = min(100.0, 100.0 * done / total)
        sys.stdout.write(f"\r  {pct:5.1f}%  {done / 1024**2:8.0f} / {total / 1024**2:.0f} MB")
        sys.stdout.flush()


def fetch(url, zip_dir, extract_dir):
    zip_path = zip_dir / url.rsplit("/", 1)[1]
    marker = extract_dir / f".{zip_path.stem}.done"
    if marker.exists():
        print(f"[skip] {zip_path.name} already extracted")
        return
    if not zip_path.exists():
        print(f"[download] {url}")
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp, _progress)
        tmp.rename(zip_path)
        print()
    print(f"[extract] {zip_path.name} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    marker.touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=sorted(FILES))
    ap.add_argument("--root", default="data")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    for name in args.datasets:
        extract_dir = Path(args.root) / name
        zip_dir = Path(args.root) / "_downloads"
        extract_dir.mkdir(parents=True, exist_ok=True)
        zip_dir.mkdir(parents=True, exist_ok=True)
        for url in FILES[name]:
            fetch(url, zip_dir, extract_dir)
            if not args.keep_zips:
                (zip_dir / url.rsplit("/", 1)[1]).unlink(missing_ok=True)
    print("Done.")


if __name__ == "__main__":
    main()
