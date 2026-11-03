# Track A — Cross-dataset evaluation of 2D object detectors (COCO → KITTI)

Zero-shot evaluation of COCO-pretrained detectors from several families on COCO val2017 (in-domain)
and KITTI 2D object (driving domain), with COCO-style metrics, latency/memory benchmarks and failure analysis.

| Family | Models (registry name) | Framework |
|---|---|---|
| YOLO (one-stage, NMS) | `yolo11n/s/m/l`, `yolov8s` | Ultralytics |
| DETR (transformer, NMS-free) | `rtdetr-l`, `rtdetr-x` | Ultralytics |
| R-CNN (two-stage) | `fasterrcnn_r50_fpn_v2` | torchvision |
| RetinaNet / FCOS (dense one-stage) | `retinanet_r50_fpn_v2`, `fcos_r50_fpn` | torchvision |
| Open-vocabulary | `yolo-world-s/l`, `gdino-tiny`, `gdino-base` | Ultralytics / HF transformers |

## Files

| File | Purpose |
|---|---|
| `env_info.py` | Hardware + package versions → `results/env.json`, `results/pip_freeze.txt` |
| `download_data.py` | Download + extract COCO val2017 and KITTI |
| `kitti_to_coco.py` | KITTI labels → COCO JSON with COCO category ids; seeded train/val split |
| `configs/kitti_mapping*.json` | Taxonomy mapping (strict default + loose ablation) |
| `detectors.py` | One `predict()` interface over Ultralytics / torchvision / HF models |
| `evaluate.py` | Inference + COCOeval: AP, AP50/75, AP S/M/L, AR, per-class and per-class-per-scale AP |
| `benchmark.py` | Batch-1 end-to-end latency (median/p90), FPS, peak GPU memory, params |
| `visualize.py` | Worst-case images (FN red / FP orange / TP green) + miss breakdown by box height & occlusion |
| `summarize.py` | All results → `results/summary.md` tables (accuracy, per-class, COCO→KITTI drop, speed, configs) |
| `run_experiments.py` | Runs the whole matrix; every command is logged to `results/command_log.txt` |

## 1. Setup — laptop (Windows, PowerShell)

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
```bash
pip install -r requirements.txt
```

**Before benchmarking:** plug in the charger and switch the laptop to its Performance/Turbo mode in
the manufacturer's control app (the "Silent" profile throttles the CPU and roughly halves detector throughput).
Record the mode and the GPU power limit (`nvidia-smi -q -d POWER`, "Current Power Limit") in the report.

## 1b. Setup — Google Colab

```python
!git clone <your-repo-url> trackA && cd trackA
%cd trackA
!pip install -q -r requirements.txt   # Colab already ships a CUDA build of torch
!python env_info.py
```
Colab GPUs differ between sessions, so use Colab for evaluation/fine-tuning, and take the **latency
numbers from the laptop only** so they're comparable.

## 2. Data

```bash
python download_data.py coco kitti
```
```bash
python kitti_to_coco.py --mapping configs/kitti_mapping.json
```
```bash
python kitti_to_coco.py --mapping configs/kitti_mapping_loose.json --suffix loose
```

Expected layout:
```
data/coco/val2017/*.jpg
data/coco/annotations/instances_val2017.json
data/kitti/training/image_2/*.png  data/kitti/training/label_2/*.txt
data/kitti/annotations/kitti_val_coco.json   (1,496 images: seed 42, 20% of the official training set)
```

### KITTI → COCO taxonomy mapping (put this table in the report)

| KITTI class | Strict (default) | Loose (ablation) |
|---|---|---|
| Car | car | car |
| Van | ignore region for *car* | car |
| Truck | truck | truck |
| Pedestrian | person | person |
| Person_sitting | ignore region for *person* | person |
| Cyclist | ignore region for *person* | person |
| Tram, Misc, DontCare | ignore region (all classes) | ignore region (all classes) |

"Ignore region" = COCO `iscrowd=1` box: detections inside it are neither TP nor FP. Detections of COCO classes
outside {person, car, truck} (e.g. *bus*, *bicycle*) are not scored on KITTI. The strict mapping
follows the official KITTI protocol, where Van and Person_sitting are "don't care" for Car and Pedestrian.

## 3. Run

```bash
python run_experiments.py --quick
```
```bash
python run_experiments.py
```

Or individual commands, e.g.
```bash
python evaluate.py --model rtdetr-l --dataset kitti --imgsz 1280 --run-name rtdetr-l_1280
```
```bash
python benchmark.py --model yolo11s --dataset kitti --half
```
```bash
python visualize.py --dataset kitti --run yolo11s --worst 12
```
```bash
python summarize.py
```

## Protocol details (for the reproducibility section)

* **mAP:** pycocotools COCOeval (bbox), conf threshold 0.001, max 100 detections/image. NMS IoU uses each
  framework's default (Ultralytics 0.7; torchvision Faster R-CNN/RetinaNet 0.5, FCOS 0.6). RT-DETR and
  Grounding DINO are NMS-free. All effective values are written to each `metrics.json`.
* **Input resolution:** Ultralytics = letterbox, long side 640 (`--imgsz`); torchvision = short side 800,
  long side ≤ 1333; Grounding DINO = HF processor default (800/1333). KITTI frames are 1242×375.
* **Latency:** batch 1, end-to-end (pre-processing + forward + post-processing/NMS), images preloaded in RAM,
  30 warm-up iterations, 200 timed, `torch.cuda.synchronize()` around every call, median + p90 reported,
  conf 0.25. Memory = `torch.cuda.max_memory_allocated` (excludes the CUDA context).
* **Grounding DINO on COCO:** the 80 class names are split into 4 prompts of 20 (256-token text limit),
  run as one batch of 4. Class score = mean sigmoid over the class name's tokens, top-100 over
  queries × classes (the official Grounding DINO COCO-eval scoring; HF's `post_process_grounded_object_detection`
  is not used because it decodes a label string per box and took ~7 s/image at conf 0.001).
  `run_experiments.py` limits it to the first 1,000 val2017 images because of the cost (~1–3 s/image).
* **KITTI split:** no public test labels, so a seeded (42) 80/20 split of the 7,481 training images
  is used; image lists are written to `data/kitti/annotations/kitti_{train,val}_images.txt`.

## What to submit alongside the PDF

`results/command_log.txt`, `results/env.json`, `results/pip_freeze.txt`, `results/summary.md`,
every `results/*/*/metrics.json`, `results/bench/*.json`, `configs/`, and this code.
