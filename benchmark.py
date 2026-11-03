"""Measure latency / FPS / peak GPU memory, batch size 1, end-to-end (pre-process + forward +
post-process/NMS), images pre-loaded in RAM so disk I/O is excluded.

Uses a deployment-style confidence threshold (0.25) rather than the 0.001 used for mAP, since
the low threshold inflates NMS time for NMS-based detectors.

Usage:
    python benchmark.py --model yolo11s --dataset kitti
    python benchmark.py --model rtdetr-l --dataset coco --half
"""
import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO

from common import add_dataset_args, command_string, dump_json, load_rgb, log_command, resolve_dataset
from detectors import REGISTRY, build_detector
from env_info import collect_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY))
    ap.add_argument("--weights")
    add_dataset_args(ap)
    ap.add_argument("--n", type=int, default=200, help="timed iterations (cycles through the images if fewer)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--tv-min-size", type=int, default=None)
    ap.add_argument("--tv-max-size", type=int, default=None)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--eval-run", default=None,
                    help="evaluate.py run name whose AP is paired with this timing in summarize.py (default: model)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    log_command(args.out)

    ann_path, img_dir = resolve_dataset(args)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_path))
    img_ids = sorted(gt.getImgIds())[:args.n]
    categories = [{"id": c, "name": gt.cats[c]["name"]} for c in sorted(gt.getCatIds())]
    images = [load_rgb(img_dir / gt.imgs[i]["file_name"]) for i in img_ids]

    torch.cuda.reset_peak_memory_stats()
    det = build_detector(args.model, device=args.device, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                         max_det=args.max_det, half=args.half, categories=categories, weights=args.weights,
                         tv_min_size=args.tv_min_size, tv_max_size=args.tv_max_size)
    for i in range(args.warmup):
        det.predict(images[i % len(images)])

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(args.n):
        img = images[i % len(images)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        det.predict(img)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)

    result = {
        "command": command_string(),
        "dataset": args.dataset,
        "eval_run": args.eval_run or args.model,
        "num_images": len(images),
        "timed_iterations": args.n,
        "image_size_hw_example": list(images[0].shape[:2]),
        "model": det.config,
        "latency_ms": {"median": float(np.median(t)), "mean": float(t.mean()), "p90": float(np.percentile(t, 90)),
                       "min": float(t.min()), "std": float(t.std())},
        "fps_from_median": float(1000 / np.median(t)),
        "peak_gpu_mem_allocated_MB": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_gpu_mem_reserved_MB": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "notes": "batch=1, end-to-end incl. pre/post-processing, images preloaded; torch allocator stats "
                 "exclude the CUDA context (~300-500 MB)",
        "env": collect_env(),
    }
    run_name = args.run_name or (args.model + ("_fp16" if args.half else ""))
    out_path = Path(args.out) / "bench" / f"{args.dataset}_{run_name}.json"
    dump_json(result, out_path)
    print(f"{args.model} on {args.dataset}: median {np.median(t):.1f} ms ({1000 / np.median(t):.1f} FPS), "
          f"p90 {np.percentile(t, 90):.1f} ms, peak alloc {result['peak_gpu_mem_allocated_MB']} MB, "
          f"{det.config['params_M']} M params -> {out_path}")


if __name__ == "__main__":
    main()
