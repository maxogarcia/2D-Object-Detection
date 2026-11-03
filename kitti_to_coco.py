"""Convert KITTI 2D object labels (training/label_2) to COCO-format JSON with COCO category ids.

KITTI has no public test labels, so we make a deterministic held-out split of the 7,481
training images (seed 42, 20% -> val). Zero-shot evaluation uses the val split; the train
split is only needed if you fine-tune.

Usage:
    python kitti_to_coco.py --kitti data/kitti --mapping configs/kitti_mapping.json
    python kitti_to_coco.py --kitti data/kitti --mapping configs/kitti_mapping_loose.json --suffix loose
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

KITTI_FIELDS = ["type", "truncated", "occluded", "alpha", "x1", "y1", "x2", "y2"]


def parse_label_file(path):
    objs = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        o = dict(zip(KITTI_FIELDS, parts[:8]))
        for k in KITTI_FIELDS[1:]:
            o[k] = float(o[k])
        objs.append(o)
    return objs


def resolve(target, name_to_id):
    """Return list of (category_id, iscrowd) annotations to emit for one KITTI object."""
    if target.startswith("ignore"):
        if ":" in target:
            return [(name_to_id[target.split(":", 1)[1]], 1)]
        return [(cid, 1) for cid in name_to_id.values()]
    return [(name_to_id[target], 0)]


def build_split(stems, args, mapping, name_to_id, split_name):
    images, annotations, stats = [], [], Counter()
    ann_id = 1
    img_dir = Path(args.kitti) / "training" / "image_2"
    lbl_dir = Path(args.kitti) / "training" / "label_2"
    for img_id, stem in enumerate(stems, start=1):
        w, h = Image.open(img_dir / f"{stem}.png").size
        images.append({"id": img_id, "file_name": f"{stem}.png", "width": w, "height": h})
        for o in parse_label_file(lbl_dir / f"{stem}.txt"):
            target = mapping["map"].get(o["type"], "ignore")
            x1, y1 = max(0.0, o["x1"]), max(0.0, o["y1"])
            x2, y2 = min(float(w), o["x2"]), min(float(h), o["y2"])
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            for cid, crowd in resolve(target, name_to_id):
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cid,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2), "iscrowd": crowd,
                    "kitti_type": o["type"], "truncated": o["truncated"], "occluded": int(o["occluded"]),
                })
                ann_id += 1
            stats[f"{o['type']} -> {target}"] += 1
    return {
        "info": {
            "description": f"KITTI 2D object, {split_name} split (seed {args.seed}, val_frac {args.val_frac}) "
                           f"of the official training set, mapped with {Path(args.mapping).name}",
            "split": split_name,
            "mapping_file": Path(args.mapping).name,
        },
        "images": images,
        "annotations": annotations,
        "categories": mapping["evaluated_categories"],
    }, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti", default="data/kitti")
    ap.add_argument("--mapping", default="configs/kitti_mapping.json")
    ap.add_argument("--out", default=None, help="default: <kitti>/annotations")
    ap.add_argument("--suffix", default="", help="appended to output file names, e.g. 'loose'")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    name_to_id = {c["name"]: c["id"] for c in mapping["evaluated_categories"]}
    stems = sorted(p.stem for p in (Path(args.kitti) / "training" / "label_2").glob("*.txt"))
    if not stems:
        raise SystemExit(f"No label files found under {args.kitti}/training/label_2")
    rng = random.Random(args.seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n_val = round(len(stems) * args.val_frac)
    splits = {"val": sorted(shuffled[:n_val]), "train": sorted(shuffled[n_val:])}

    out_dir = Path(args.out or Path(args.kitti) / "annotations")
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.suffix}" if args.suffix else ""
    for split_name, split_stems in splits.items():
        if not split_stems:
            continue
        coco, stats = build_split(split_stems, args, mapping, name_to_id, split_name)
        out_path = out_dir / f"kitti_{split_name}{sfx}_coco.json"
        out_path.write_text(json.dumps(coco))
        (out_dir / f"kitti_{split_name}_images.txt").write_text("\n".join(split_stems) + "\n")
        n_eval = sum(1 for a in coco["annotations"] if not a["iscrowd"])
        print(f"{split_name}: {len(split_stems)} images, {n_eval} evaluated boxes -> {out_path}")
        for k, v in sorted(stats.items()):
            print(f"    {k:35s} {v}")


if __name__ == "__main__":
    main()
