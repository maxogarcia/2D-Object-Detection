# Results summary

AP values are COCO-style, x100.

## Accuracy

| dataset | run | family | #img | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 |
|---|---|---|---|---|---|---|---|---|---|---|
| coco | fasterrcnn_r50_fpn_v2 | R-CNN | 200 | 52.6 | 72.3 | 58.4 | 38.3 | 58.4 | 70.6 | 66.0 |
| coco | fcos_r50_fpn | RetinaNet/FCOS | 200 | 43.9 | 62.7 | 48.2 | 30.4 | 48.6 | 61.8 | 61.7 |
| coco | gdino-tiny | Open-vocab | 200 | 52.6 | 67.0 | 56.7 | 36.6 | 60.9 | 69.6 | 72.7 |
| coco | rtdetr-l | DETR | 200 | 57.3 | 75.9 | 61.3 | 41.9 | 60.3 | 76.0 | 72.9 |
| coco | yolo-world-s | Open-vocab | 200 | 41.6 | 56.9 | 44.9 | 27.5 | 49.7 | 60.2 | 56.5 |
| coco | yolo11s | YOLO | 200 | 49.6 | 65.2 | 54.6 | 33.5 | 54.0 | 71.7 | 63.9 |
| kitti | fasterrcnn_r50_fpn_v2 | R-CNN | 200 | 31.5 | 56.1 | 31.7 | 24.3 | 28.6 | 50.0 | 54.2 |
| kitti | fcos_r50_fpn | RetinaNet/FCOS | 200 | 31.6 | 55.7 | 32.8 | 34.7 | 27.5 | 49.6 | 52.9 |
| kitti | gdino-tiny | Open-vocab | 200 | 39.1 | 62.9 | 41.4 | 38.9 | 33.9 | 59.8 | 58.7 |
| kitti | rtdetr-l | DETR | 200 | 32.5 | 55.5 | 34.3 | 29.3 | 32.1 | 48.8 | 52.9 |
| kitti | rtdetr-l_1280 | DETR | 200 | 33.8 | 56.9 | 36.1 | 30.9 | 33.8 | 48.6 | 56.0 |
| kitti | yolo-world-s | Open-vocab | 200 | 35.3 | 58.1 | 38.7 | 38.7 | 29.6 | 58.6 | 53.1 |
| kitti | yolo11s | YOLO | 200 | 28.6 | 50.8 | 29.6 | 25.3 | 26.9 | 46.5 | 42.1 |
| kitti | yolo11s_1280 | YOLO | 200 | 31.6 | 55.2 | 33.7 | 27.4 | 31.4 | 46.7 | 48.2 |
| kitti-loose | fasterrcnn_r50_fpn_v2 | R-CNN | 200 | 28.3 | 51.6 | 28.4 | 22.3 | 25.6 | 44.2 | 52.3 |
| kitti-loose | yolo11s | YOLO | 200 | 25.4 | 47.2 | 25.7 | 24.6 | 23.7 | 39.3 | 39.6 |

## Per-class AP - coco (15 most frequent classes)

| run | person | chair | cup | car | book | wine glass | bottle | bowl | dining table | backpack | bus | tie | motorcycle | handbag | knife |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasterrcnn_r50_fpn_v2 | 60.3 | 34.9 | 59.1 | 57.5 | 12.1 | 40.3 | 37.1 | 44.8 | 57.2 | 18.7 | 65.7 | 47.6 | 60.4 | 23.7 | 40.1 |
| fcos_r50_fpn | 53.4 | 28.2 | 51.8 | 50.6 | 10.9 | 30.3 | 28.1 | 42.2 | 48.3 | 15.0 | 42.7 | 31.9 | 55.8 | 15.0 | 26.2 |
| gdino-tiny | 64.7 | 41.7 | 64.1 | 53.1 | 5.4 | 53.9 | 40.6 | 44.6 | 16.3 | 28.1 | 69.1 | 51.3 | 73.1 | 16.1 | 62.8 |
| rtdetr-l | 63.2 | 41.2 | 63.6 | 57.5 | 18.3 | 46.8 | 34.7 | 50.6 | 65.5 | 18.0 | 62.4 | 44.9 | 72.4 | 31.1 | 58.9 |
| yolo-world-s | 56.8 | 35.7 | 55.9 | 39.7 | 1.5 | 38.6 | 23.6 | 31.8 | 56.5 | 11.7 | 48.3 | 33.3 | 57.5 | 13.6 | 22.6 |
| yolo11s | 59.2 | 39.7 | 54.0 | 47.0 | 9.7 | 37.2 | 22.5 | 38.5 | 61.5 | 15.9 | 51.2 | 36.0 | 68.0 | 24.7 | 36.9 |

## Per-class AP - kitti

| run | person | car | truck |
|---|---|---|---|
| fasterrcnn_r50_fpn_v2 | 28.4 | 50.8 | 15.3 |
| fcos_r50_fpn | 26.7 | 49.4 | 18.7 |
| gdino-tiny | 29.4 | 51.0 | 37.0 |
| rtdetr-l | 26.2 | 51.4 | 19.8 |
| rtdetr-l_1280 | 30.4 | 51.2 | 19.8 |
| yolo-world-s | 24.3 | 48.2 | 33.4 |
| yolo11s | 25.4 | 47.3 | 13.2 |
| yolo11s_1280 | 27.9 | 51.4 | 15.4 |

## Per-class AP - kitti-loose

| run | person | car | truck |
|---|---|---|---|
| fasterrcnn_r50_fpn_v2 | 21.1 | 48.4 | 15.3 |
| yolo11s | 17.7 | 45.4 | 13.2 |

## Cross-dataset: fasterrcnn_r50_fpn_v2 (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 87 | 60.3 -> 28.4 | 40.6 -> 14.3 | 67.7 -> 28.3 | 76.8 -> 58.9 |
| car | 712 | 57.5 -> 50.8 | 50.0 -> 43.6 | 70.7 -> 49.1 | 72.2 -> 57.5 |
| truck | 26 | 38.9 -> 15.3 | 42.2 -> 14.9 | 40.2 -> 8.3 | 80.1 -> 33.5 |
| **all** |  | 52.6 -> 31.5 | 38.3 -> 24.3 | 58.4 -> 28.6 | 70.6 -> 50.0 |

## Cross-dataset: fcos_r50_fpn (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 87 | 53.4 -> 26.7 | 36.1 -> 5.4 | 61.1 -> 27.8 | 68.6 -> 59.0 |
| car | 712 | 50.6 -> 49.4 | 41.9 -> 43.1 | 65.2 -> 47.2 | 67.3 -> 57.0 |
| truck | 26 | 19.5 -> 18.7 | 21.6 -> 55.5 | 18.6 -> 7.4 | 51.7 -> 32.9 |
| **all** |  | 43.9 -> 31.6 | 30.4 -> 34.7 | 48.6 -> 27.5 | 61.8 -> 49.6 |

## Cross-dataset: gdino-tiny (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 87 | 64.7 -> 29.4 | 41.4 -> 13.7 | 73.9 -> 30.4 | 83.8 -> 55.8 |
| car | 712 | 53.1 -> 51.0 | 41.5 -> 51.0 | 68.8 -> 50.9 | 75.7 -> 55.8 |
| truck | 26 | 31.7 -> 37.0 | 15.9 -> 51.9 | 39.5 -> 20.5 | 83.5 -> 67.7 |
| **all** |  | 52.6 -> 39.1 | 36.6 -> 38.9 | 60.9 -> 33.9 | 69.6 -> 59.8 |

## Cross-dataset: rtdetr-l (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 87 | 63.2 -> 26.2 | 40.5 -> 5.1 | 69.2 -> 29.8 | 85.3 -> 57.2 |
| car | 712 | 57.5 -> 51.4 | 49.1 -> 49.1 | 69.5 -> 49.1 | 85.2 -> 58.6 |
| truck | 26 | 32.8 -> 19.8 | 24.5 -> 33.6 | 37.9 -> 17.5 | 75.0 -> 30.5 |
| **all** |  | 57.3 -> 32.5 | 41.9 -> 29.3 | 60.3 -> 32.1 | 76.0 -> 48.8 |

## Cross-dataset: yolo-world-s (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 87 | 56.8 -> 24.3 | 31.4 -> 2.4 | 67.4 -> 27.2 | 79.1 -> 57.7 |
| car | 712 | 39.7 -> 48.2 | 33.1 -> 35.1 | 51.6 -> 46.1 | 56.8 -> 59.5 |
| truck | 26 | 24.6 -> 33.4 | 24.0 -> 78.8 | 18.7 -> 15.4 | 92.5 -> 58.7 |
| **all** |  | 41.6 -> 35.3 | 27.5 -> 38.7 | 49.7 -> 29.6 | 60.2 -> 58.6 |

## Cross-dataset: yolo11s (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 87 | 59.2 -> 25.4 | 36.6 -> 4.2 | 67.1 -> 28.3 | 81.4 -> 57.6 |
| car | 712 | 47.0 -> 47.3 | 36.1 -> 36.0 | 66.1 -> 45.8 | 64.1 -> 56.5 |
| truck | 26 | 34.5 -> 13.2 | 24.4 -> 35.6 | 32.7 -> 6.6 | 95.0 -> 25.6 |
| **all** |  | 49.6 -> 28.6 | 33.5 -> 25.3 | 54.0 -> 26.9 | 71.7 -> 46.5 |

## Cross-dataset: fasterrcnn_r50_fpn_v2 (COCO -> kitti-loose)

| class | #gt (kitti-loose) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 138 | 60.3 -> 21.1 | 40.6 -> 12.0 | 67.7 -> 21.4 | 76.8 -> 42.1 |
| car | 795 | 57.5 -> 48.4 | 50.0 -> 40.1 | 70.7 -> 47.0 | 72.2 -> 57.1 |
| truck | 26 | 38.9 -> 15.3 | 42.2 -> 14.9 | 40.2 -> 8.3 | 80.1 -> 33.5 |
| **all** |  | 52.6 -> 28.3 | 38.3 -> 22.3 | 58.4 -> 25.6 | 70.6 -> 44.2 |

## Cross-dataset: yolo11s (COCO -> kitti-loose)

| class | #gt (kitti-loose) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 138 | 59.2 -> 17.7 | 36.6 -> 4.0 | 67.1 -> 21.0 | 81.4 -> 36.6 |
| car | 795 | 47.0 -> 45.4 | 36.1 -> 34.1 | 66.1 -> 43.4 | 64.1 -> 55.8 |
| truck | 26 | 34.5 -> 13.2 | 24.4 -> 35.6 | 32.7 -> 6.6 | 95.0 -> 25.6 |
| **all** |  | 49.6 -> 25.4 | 33.5 -> 24.6 | 54.0 -> 23.7 | 71.7 -> 39.3 |

## Configurations

| dataset | run | checkpoint | input | conf | NMS IoU | framework | command |
|---|---|---|---|---|---|---|---|
| coco | fasterrcnn_r50_fpn_v2 | FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.5 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fasterrcnn_r50_fpn_v2 --dataset coco --max-images 200` |
| coco | fcos_r50_fpn | FCOS_ResNet50_FPN_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.6 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fcos_r50_fpn --dataset coco --max-images 200` |
| coco | gdino-tiny | IDEA-Research/grounding-dino-tiny | processor default (short side 800, long side <= 1333) | 0.001 | none | transformers 5.21.0 | `python.exe evaluate.py --model gdino-tiny --dataset coco --max-images 200` |
| coco | rtdetr-l | rtdetr-l.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | none (set prediction, NMS-free) | ultralytics 8.4.190 | `python.exe evaluate.py --model rtdetr-l --dataset coco --max-images 200` |
| coco | yolo-world-s | yolov8s-worldv2.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo-world-s --dataset coco --max-images 200` |
| coco | yolo11s | yolo11s.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo11s --dataset coco --max-images 200` |
| kitti | fasterrcnn_r50_fpn_v2 | FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.5 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fasterrcnn_r50_fpn_v2 --dataset kitti --max-images 200` |
| kitti | fcos_r50_fpn | FCOS_ResNet50_FPN_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.6 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fcos_r50_fpn --dataset kitti --max-images 200` |
| kitti | gdino-tiny | IDEA-Research/grounding-dino-tiny | processor default (short side 800, long side <= 1333) | 0.001 | none | transformers 5.21.0 | `python.exe evaluate.py --model gdino-tiny --dataset kitti --max-images 200` |
| kitti | rtdetr-l | rtdetr-l.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | none (set prediction, NMS-free) | ultralytics 8.4.190 | `python.exe evaluate.py --model rtdetr-l --dataset kitti --max-images 200` |
| kitti | rtdetr-l_1280 | rtdetr-l.pt | letterbox, long side 1280 (rect, stride-padded) | 0.001 | none (set prediction, NMS-free) | ultralytics 8.4.190 | `python.exe evaluate.py --model rtdetr-l --dataset kitti --imgsz 1280 --run-name rtdetr-l_1280 --max-images 200` |
| kitti | yolo-world-s | yolov8s-worldv2.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo-world-s --dataset kitti --max-images 200` |
| kitti | yolo11s | yolo11s.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo11s --dataset kitti --max-images 200` |
| kitti | yolo11s_1280 | yolo11s.pt | letterbox, long side 1280 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo11s --dataset kitti --imgsz 1280 --run-name yolo11s_1280 --max-images 200` |
| kitti-loose | fasterrcnn_r50_fpn_v2 | FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.5 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fasterrcnn_r50_fpn_v2 --dataset kitti-loose --max-images 200` |
| kitti-loose | yolo11s | yolo11s.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo11s --dataset kitti-loose --max-images 200` |
