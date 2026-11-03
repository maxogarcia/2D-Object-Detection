"""Failure analysis: find the images where a model misses the most GT boxes and draw them.

Green = ground truth matched, red = missed GT (false negative), orange = false positive,
grey dashed = ignore/crowd regions. Matching: greedy by score, same class, IoU >= 0.5.

Usage:
    python visualize.py --dataset kitti --run yolo11s --worst 12
    python visualize.py --dataset kitti --run yolo11s --image-ids 17 42 --score 0.3
"""
import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from common import add_dataset_args, dump_json, resolve_dataset


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def match_image(gts, dets, cat_ids, iou_thr=0.5):
    """Returns (matched_gt, missed_gt, fp_dets, crowd_gt)."""
    crowd = [g for g in gts if g["iscrowd"]]
    real = [g for g in gts if not g["iscrowd"]]
    used, fps = set(), []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if d["category_id"] not in cat_ids:
            continue
        best, best_iou = None, iou_thr
        for i, g in enumerate(real):
            if i not in used and g["category_id"] == d["category_id"]:
                v = iou_xywh(d["bbox"], g["bbox"])
                if v >= best_iou:
                    best, best_iou = i, v
        if best is not None:
            used.add(best)
        elif not any(c["category_id"] == d["category_id"] and iou_xywh(d["bbox"], c["bbox"]) > 0 for c in crowd):
            fps.append(d)
    matched = [g for i, g in enumerate(real) if i in used]
    missed = [g for i, g in enumerate(real) if i not in used]
    return matched, missed, fps, crowd


def draw(img_path, matched, missed, fps, crowd, names, out_path):
    im = Image.open(img_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    rect = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]
    for g in crowd:
        dr.rectangle(rect(g["bbox"]), outline=(150, 150, 150), width=1)
    for g in matched:
        dr.rectangle(rect(g["bbox"]), outline=(0, 200, 0), width=2)
    for g in missed:
        dr.rectangle(rect(g["bbox"]), outline=(230, 0, 0), width=3)
        dr.text((g["bbox"][0], g["bbox"][1] - 10), f"FN {names[g['category_id']]}", fill=(230, 0, 0))
    for d in fps:
        dr.rectangle(rect(d["bbox"]), outline=(255, 150, 0), width=2)
        dr.text((d["bbox"][0], d["bbox"][1] + 2), f"FP {names[d['category_id']]} {d['score']:.2f}", fill=(255, 150, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    add_dataset_args(ap)
    ap.add_argument("--run", required=True, help="run name under results/<dataset>/")
    ap.add_argument("--score", type=float, default=0.3, help="detections below this are dropped before matching")
    ap.add_argument("--worst", type=int, default=12, help="draw the N images with most missed GT")
    ap.add_argument("--image-ids", type=int, nargs="*", help="draw these image ids instead")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ann_path, img_dir = resolve_dataset(args)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_path))
    run_dir = Path(args.out) / args.dataset / args.run
    dets = json.loads((run_dir / "detections.json").read_text())
    names = {c["id"]: c["name"] for c in gt.dataset["categories"]}
    cat_ids = set(names)
    by_img = defaultdict(list)
    for d in dets:
        if d["score"] >= args.score:
            by_img[d["image_id"]].append(d)
    # same image subset evaluate.py used (first N by id), including images with zero detections
    n_eval = json.loads((run_dir / "metrics.json").read_text())["dataset"]["num_images"]
    evaluated = args.image_ids or sorted(gt.getImgIds())[:n_eval]

    per_image, totals = [], defaultdict(lambda: np.zeros(3, int))  # [tp, fn, fp] per class
    for img_id in evaluated:
        gts = gt.loadAnns(gt.getAnnIds(imgIds=[img_id]))
        m, miss, fp, crowd = match_image(gts, by_img[img_id], cat_ids)
        per_image.append((img_id, len(miss), len(fp), m, miss, fp, crowd))
        for g in m:
            totals[g["category_id"]][0] += 1
        for g in miss:
            totals[g["category_id"]][1] += 1
        for d in fp:
            totals[d["category_id"]][2] += 1

    chosen = per_image if args.image_ids else sorted(per_image, key=lambda x: (-x[1], -x[2]))[:args.worst]
    vis_dir = run_dir / "failures"
    for img_id, n_miss, n_fp, m, miss, fp, crowd in chosen:
        info = gt.imgs[img_id]
        draw(img_dir / info["file_name"], m, miss, fp, crowd, names,
             vis_dir / f"{img_id:06d}_fn{n_miss}_fp{n_fp}.png")

    # Size/occlusion breakdown of misses, useful for the failure-analysis section
    missed_all = [g for x in per_image for g in x[4]]
    def bucket(g):
        h = g["bbox"][3]
        return "h<25px" if h < 25 else "25-40px" if h < 40 else "40-100px" if h < 100 else ">=100px"
    summary = {
        "score_threshold": args.score, "iou_threshold": 0.5,
        "per_class": {names[c]: {"TP": int(v[0]), "FN": int(v[1]), "FP": int(v[2]),
                                 "recall": round(v[0] / max(1, v[0] + v[1]), 3),
                                 "precision": round(v[0] / max(1, v[0] + v[2]), 3)} for c, v in totals.items()},
        "missed_by_box_height": {b: sum(1 for g in missed_all if bucket(g) == b)
                                 for b in ["h<25px", "25-40px", "40-100px", ">=100px"]},
        "missed_by_occlusion": {str(k): sum(1 for g in missed_all if g.get("occluded") == k) for k in (0, 1, 2, 3)}
        if missed_all and "occluded" in missed_all[0] else None,
        "drawn_images": [f"{x[0]:06d}" for x in chosen],
    }
    dump_json(summary, vis_dir / "failure_summary.json")
    print(json.dumps(summary, indent=2))
    print(f"\nDrew {len(chosen)} images into {vis_dir}")


if __name__ == "__main__":
    main()
