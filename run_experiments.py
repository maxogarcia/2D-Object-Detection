"""Run the full Track A experiment matrix (evaluation -> benchmarks -> failure images -> summary).

Every underlying command is appended to results/command_log.txt, which you submit with the report.

Usage:
    python run_experiments.py --dry-run              # print the commands only
    python run_experiments.py --quick                # 200 images per dataset, sanity check (~minutes)
    python run_experiments.py                        # full run
    python run_experiments.py --stage bench          # only latency/memory
    python run_experiments.py --models yolo11s rtdetr-l --datasets kitti
"""
import argparse
import subprocess
import sys

# Core: 3 configurations from 3 families (meets the ">= 3 configs, >= 2 families" requirement).
CORE = ["yolo11s", "rtdetr-l", "fasterrcnn_r50_fpn_v2"]
# Optional extras: 4th closed-set family + open-vocabulary (enables the closed vs open-vocab analysis).
EXTRA = ["fcos_r50_fpn", "yolo-world-s", "gdino-tiny"]
DATASETS = ["coco", "kitti"]

# Grounding DINO runs 4 text-prompt passes per COCO image (80 classes / 20 per chunk), so cap it.
MAX_IMAGES = {("gdino-tiny", "coco"): 1000}

# Ablations that feed the technical analyses.
ABLATIONS = [
    # taxonomy ablation: how much of the KITTI drop is label-definition mismatch?
    ["--model", "yolo11s", "--dataset", "kitti-loose"],
    ["--model", "fasterrcnn_r50_fpn_v2", "--dataset", "kitti-loose"],
    # resolution ablation: KITTI is 1242x375, 640 letterbox shrinks small objects ~2x
    ["--model", "yolo11s", "--dataset", "kitti", "--imgsz", "1280", "--run-name", "yolo11s_1280"],
    ["--model", "rtdetr-l", "--dataset", "kitti", "--imgsz", "1280", "--run-name", "rtdetr-l_1280"],
]


def run(cmd, dry):
    print(">>", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["eval", "ablation", "bench", "vis", "summary", "all"], default="all")
    ap.add_argument("--models", nargs="+", default=CORE + EXTRA)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--quick", action="store_true", help="200 images per eval, 50 timed iterations")
    ap.add_argument("--half", action="store_true", help="also benchmark FP16")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    stages = ["eval", "ablation", "bench", "vis", "summary"] if args.stage == "all" else [args.stage]

    if "eval" in stages:
        run([py, "env_info.py"], args.dry_run)
        for ds in args.datasets:
            for m in args.models:
                cmd = [py, "evaluate.py", "--model", m, "--dataset", ds]
                n = 200 if args.quick else MAX_IMAGES.get((m, ds))
                if n:
                    cmd += ["--max-images", str(n)]
                run(cmd, args.dry_run)

    if "ablation" in stages:
        for extra in ABLATIONS:
            cmd = [py, "evaluate.py", *extra] + (["--max-images", "200"] if args.quick else [])
            run(cmd, args.dry_run)

    if "bench" in stages:
        n = ["--n", "50"] if args.quick else []
        for ds in args.datasets:
            for m in args.models:
                run([py, "benchmark.py", "--model", m, "--dataset", ds, *n], args.dry_run)
                if args.half and not m.startswith("gdino"):
                    run([py, "benchmark.py", "--model", m, "--dataset", ds, "--half", *n], args.dry_run)
        run([py, "benchmark.py", "--model", "yolo11s", "--dataset", "kitti", "--imgsz", "1280",
             "--run-name", "yolo11s_1280", "--eval-run", "yolo11s_1280", *n], args.dry_run)

    if "vis" in stages:
        for m in args.models:
            run([py, "visualize.py", "--dataset", "kitti", "--run", m, "--worst", "12"], args.dry_run)

    if "summary" in stages:
        run([py, "summarize.py"], args.dry_run)


if __name__ == "__main__":
    main()
