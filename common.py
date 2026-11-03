"""Dataset registry and small helpers shared by evaluate.py, benchmark.py and visualize.py."""
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path("data")

# name -> (annotation json, image dir)
DATASETS = {
    "coco": ("coco/annotations/instances_val2017.json", "coco/val2017"),
    "kitti": ("kitti/annotations/kitti_val_coco.json", "kitti/training/image_2"),
    "kitti-loose": ("kitti/annotations/kitti_val_loose_coco.json", "kitti/training/image_2"),
}


def add_dataset_args(ap):
    ap.add_argument("--dataset", default="coco",
                    help=f"one of {list(DATASETS)} (or use --ann/--images for a custom COCO-format set)")
    ap.add_argument("--ann", help="override: COCO-format annotation json")
    ap.add_argument("--images", help="override: image directory")
    ap.add_argument("--data-root", default=str(DATA_ROOT))


def resolve_dataset(args):
    ann, img = DATASETS.get(args.dataset, (None, None))
    ann = Path(args.ann) if args.ann else Path(args.data_root) / ann
    img = Path(args.images) if args.images else Path(args.data_root) / img
    return ann, img


def load_rgb(path):
    """Load as RGB uint8 HxWx3. No EXIF rotation, matching pycocotools/Ultralytics behaviour."""
    return np.asarray(Image.open(path).convert("RGB"))


def command_string():
    return " ".join(shlex.quote(a) for a in [Path(sys.executable).name, *sys.argv])


def log_command(results_dir="results"):
    """Append the exact command to results/command_log.txt (submit this file with the report)."""
    Path(results_dir).mkdir(exist_ok=True)
    with open(Path(results_dir) / "command_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {command_string()}\n")


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
