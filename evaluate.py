"""Run a detector over a COCO-format dataset and compute COCO metrics.

Outputs results/<dataset>/<run>/metrics.json (overall + per-class + per-class-per-scale AP, AR,
full config, environment, exact command) and detections.json (for visualize.py / re-scoring).

Usage:
    python evaluate.py --model yolo11s --dataset coco
    python evaluate.py --model fasterrcnn_r50_fpn_v2 --dataset kitti
    python evaluate.py --model rtdetr-l --dataset kitti --imgsz 1280 --run-name rtdetr-l_1280
    python evaluate.py --model gdino-tiny --dataset coco --max-images 500
"""
import argparse
import contextlib
import io
import math
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import add_dataset_args, command_string, dump_json, load_rgb, log_command, resolve_dataset
from detectors import REGISTRY, build_detector
from env_info import collect_env

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **_):
        return x

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
AREAS = ["all", "small", "medium", "large"]


def _mean_valid(a):
    a = a[a > -1]
    return float(a.mean()) if a.size else None


def per_class_metrics(E, gt, img_ids):
    precision = E.eval["precision"]  # [IoU thr, recall pts, class, area, maxDets]
    recall = E.eval["recall"]        # [IoU thr, class, area, maxDets]
    rows = []
    for k, cid in enumerate(E.params.catIds):
        anns = gt.loadAnns(gt.getAnnIds(imgIds=img_ids, catIds=[cid], iscrowd=False))
        row = {"category_id": cid, "name": gt.cats[cid]["name"], "num_gt": len(anns),
               "AP": _mean_valid(precision[:, :, k, 0, -1]),
               "AP50": _mean_valid(precision[0, :, k, 0, -1]),
               "AP75": _mean_valid(precision[5, :, k, 0, -1]),
               "AR100": _mean_valid(recall[:, k, 0, -1])}
        for a in (1, 2, 3):
            row[f"AP_{AREAS[a]}"] = _mean_valid(precision[:, :, k, a, -1])
            row[f"num_gt_{AREAS[a]}"] = sum(1 for x in anns if E.params.areaRng[a][0] <= x["area"] < E.params.areaRng[a][1])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY))
    ap.add_argument("--weights", help="override checkpoint path / id (e.g. a fine-tuned .pt)")
    add_dataset_args(ap)
    ap.add_argument("--conf", type=float, default=0.001, help="low threshold for mAP (standard)")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU (default: model's own default)")
    ap.add_argument("--imgsz", type=int, default=None, help="Ultralytics input long side (default 640)")
    ap.add_argument("--tv-min-size", type=int, default=None, help="torchvision short side (default 800)")
    ap.add_argument("--tv-max-size", type=int, default=None, help="torchvision long-side cap (default 1333)")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--half", action="store_true", help="FP16 inference")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-images", type=int, default=None, help="evaluate first N images (sorted by id)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    log_command(args.out)

    ann_path, img_dir = resolve_dataset(args)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_path))
    img_ids = sorted(gt.getImgIds())
    if args.max_images:
        img_ids = img_ids[:args.max_images]
    cat_ids = sorted(gt.getCatIds())
    categories = [{"id": c, "name": gt.cats[c]["name"]} for c in cat_ids]

    det = build_detector(args.model, device=args.device, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                         max_det=args.max_det, half=args.half, categories=categories, weights=args.weights,
                         tv_min_size=args.tv_min_size, tv_max_size=args.tv_max_size)
    run_name = args.run_name or args.model
    out_dir = Path(args.out) / args.dataset / run_name
    print(f"Evaluating {args.model} on {args.dataset}: {len(img_ids)} images -> {out_dir}")

    detections = []
    t0 = time.perf_counter()
    for img_id in tqdm(img_ids, desc=args.model):
        info = gt.imgs[img_id]
        boxes, scores, cats = det.predict(load_rgb(img_dir / info["file_name"]))
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, cats):
            detections.append({"image_id": img_id, "category_id": int(c),
                               "bbox": [round(float(x1), 2), round(float(y1), 2),
                                        round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                               "score": round(float(s), 5)})
    wall = time.perf_counter() - t0
    dump_json(detections, out_dir / "detections.json")

    metrics = {"stats": dict.fromkeys(STAT_NAMES), "per_class": []}
    if detections:
        dt = gt.loadRes(detections)
        E = COCOeval(gt, dt, "bbox")
        E.params.imgIds = img_ids
        E.params.catIds = cat_ids  # detections of classes outside the dataset taxonomy are ignored
        E.evaluate()
        E.accumulate()
        E.summarize()
        metrics["stats"] = {k: (None if math.isnan(v) or v < 0 else float(v)) for k, v in zip(STAT_NAMES, E.stats)}
        metrics["per_class"] = per_class_metrics(E, gt, img_ids)
    else:
        print("WARNING: no detections produced")

    result = {
        "command": command_string(),
        "dataset": {"name": args.dataset, "annotations": str(ann_path), "images": str(img_dir),
                    "split": gt.dataset.get("info", {}).get("description", ann_path.stem),
                    "num_images": len(img_ids), "evaluated_categories": categories},
        "model": det.config,
        "eval_wall_time_s": round(wall, 1),
        "eval_ms_per_image_incl_io": round(1000 * wall / max(1, len(img_ids)), 1),
        **metrics,
        "env": collect_env(),
    }
    dump_json(result, out_dir / "metrics.json")

    print(f"\n{'class':15s} {'#gt':>6s} {'AP':>6s} {'AP50':>6s} {'APs':>6s} {'APm':>6s} {'APl':>6s}")
    fmt = lambda v: f"{v:6.3f}" if v is not None else "   n/a"
    for r in metrics["per_class"]:
        if r["num_gt"]:
            print(f"{r['name']:15s} {r['num_gt']:6d} {fmt(r['AP'])} {fmt(r['AP50'])} "
                  f"{fmt(r['AP_small'])} {fmt(r['AP_medium'])} {fmt(r['AP_large'])}")
    print(f"\nSaved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
