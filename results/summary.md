# Results summary

AP values are COCO-style, x100.

## Accuracy

| dataset | run | family | #img | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 |
|---|---|---|---|---|---|---|---|---|---|---|
| coco | fasterrcnn_r50_fpn_v2 | R-CNN | 5000 | 46.9 | 67.9 | 51.4 | 31.0 | 50.8 | 60.4 | 63.0 |
| coco | fcos_r50_fpn | RetinaNet/FCOS | 5000 | 39.2 | 58.3 | 42.0 | 22.2 | 42.4 | 51.3 | 57.1 |
| coco | gdino-tiny | Open-vocab | 1000 | 49.9 | 64.5 | 54.3 | 34.9 | 52.9 | 64.0 | 71.6 |
| coco | rtdetr-l | DETR | 5000 | 52.6 | 70.9 | 57.0 | 33.9 | 57.3 | 70.7 | 69.0 |
| coco | yolo-world-s | Open-vocab | 5000 | 36.6 | 51.0 | 39.7 | 20.4 | 40.5 | 48.8 | 54.8 |
| coco | yolo11s | YOLO | 5000 | 45.5 | 62.0 | 49.2 | 27.6 | 50.1 | 62.5 | 60.2 |
| kitti | fasterrcnn_r50_fpn_v2 | R-CNN | 1496 | 33.8 | 61.1 | 34.3 | 20.8 | 32.6 | 45.0 | 56.1 |
| kitti | fcos_r50_fpn | RetinaNet/FCOS | 1496 | 33.5 | 60.3 | 34.3 | 28.7 | 31.5 | 44.0 | 53.6 |
| kitti | gdino-tiny | Open-vocab | 1496 | 39.7 | 67.8 | 42.4 | 35.6 | 39.0 | 47.9 | 59.8 |
| kitti | rtdetr-l | DETR | 1496 | 32.5 | 57.6 | 32.8 | 27.7 | 31.7 | 43.8 | 55.5 |
| kitti | rtdetr-l_1280 | DETR | 1496 | 34.1 | 57.2 | 35.6 | 31.2 | 34.2 | 43.3 | 58.6 |
| kitti | yolo-world-s | Open-vocab | 1496 | 35.1 | 60.6 | 37.4 | 26.9 | 33.7 | 50.2 | 55.7 |
| kitti | yolo11s | YOLO | 1496 | 28.5 | 53.5 | 27.9 | 15.8 | 26.4 | 41.6 | 44.8 |
| kitti | yolo11s_1280 | YOLO | 1496 | 32.5 | 58.4 | 33.0 | 21.0 | 33.1 | 41.5 | 50.3 |
| kitti-loose | fasterrcnn_r50_fpn_v2 | R-CNN | 1496 | 30.5 | 55.8 | 31.0 | 19.9 | 29.4 | 39.9 | 53.8 |
| kitti-loose | yolo11s | YOLO | 1496 | 25.5 | 48.8 | 24.8 | 15.6 | 23.5 | 36.3 | 42.2 |

## Per-class AP - coco (15 most frequent classes)

| run | person | car | chair | book | bottle | cup | dining table | traffic light | bowl | handbag | bird | boat | truck | bench | umbrella |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasterrcnn_r50_fpn_v2 | 59.6 | 49.9 | 33.9 | 19.5 | 44.9 | 48.5 | 30.2 | 31.5 | 47.1 | 20.5 | 41.1 | 32.5 | 42.7 | 30.6 | 45.6 |
| fcos_r50_fpn | 53.3 | 42.2 | 24.9 | 12.4 | 35.3 | 41.6 | 25.4 | 25.2 | 36.7 | 12.6 | 34.5 | 23.0 | 34.2 | 20.3 | 36.4 |
| gdino-tiny | 65.2 | 51.7 | 37.5 | 11.6 | 44.0 | 59.0 | 6.7 | 31.2 | 47.9 | 17.7 | 38.8 | 31.8 | 28.0 | 34.2 | 33.6 |
| rtdetr-l | 61.5 | 49.9 | 39.9 | 21.6 | 47.2 | 51.9 | 38.9 | 30.7 | 50.3 | 26.2 | 46.1 | 36.5 | 45.8 | 31.6 | 52.9 |
| yolo-world-s | 56.3 | 39.1 | 29.6 | 6.2 | 33.5 | 41.5 | 27.3 | 23.2 | 40.5 | 11.0 | 31.3 | 23.8 | 29.1 | 23.9 | 35.7 |
| yolo11s | 58.6 | 44.8 | 33.9 | 14.7 | 39.6 | 43.4 | 33.0 | 27.0 | 43.9 | 15.3 | 37.9 | 30.2 | 35.5 | 27.8 | 46.3 |

## Per-class AP - kitti

| run | person | car | truck |
|---|---|---|---|
| fasterrcnn_r50_fpn_v2 | 29.8 | 49.8 | 21.6 |
| fcos_r50_fpn | 27.8 | 48.9 | 23.9 |
| gdino-tiny | 30.1 | 51.0 | 38.1 |
| rtdetr-l | 28.3 | 51.4 | 17.8 |
| rtdetr-l_1280 | 29.7 | 50.9 | 21.7 |
| yolo-world-s | 24.9 | 48.1 | 32.4 |
| yolo11s | 26.7 | 47.0 | 11.9 |
| yolo11s_1280 | 28.3 | 51.1 | 18.0 |

## Per-class AP - kitti-loose

| run | person | car | truck |
|---|---|---|---|
| fasterrcnn_r50_fpn_v2 | 21.8 | 48.0 | 21.6 |
| yolo11s | 19.4 | 45.3 | 11.9 |

## Cross-dataset: fasterrcnn_r50_fpn_v2 (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 831 | 59.6 -> 29.8 | 42.5 -> 10.7 | 66.2 -> 29.9 | 75.9 -> 44.9 |
| car | 5798 | 49.9 -> 49.8 | 40.7 -> 41.4 | 62.9 -> 47.4 | 69.9 -> 57.8 |
| truck | 219 | 42.7 -> 21.6 | 25.4 -> 10.2 | 44.5 -> 20.6 | 56.4 -> 32.2 |
| **all** |  | 46.9 -> 33.8 | 31.0 -> 20.8 | 50.8 -> 32.6 | 60.4 -> 45.0 |

## Cross-dataset: fcos_r50_fpn (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 831 | 53.3 -> 27.8 | 35.3 -> 6.8 | 59.9 -> 28.1 | 71.3 -> 44.4 |
| car | 5798 | 42.2 -> 48.9 | 32.6 -> 39.3 | 56.9 -> 46.4 | 62.2 -> 57.4 |
| truck | 219 | 34.2 -> 23.9 | 16.4 -> 39.9 | 32.7 -> 20.0 | 51.0 -> 30.3 |
| **all** |  | 39.2 -> 33.5 | 22.2 -> 28.7 | 42.4 -> 31.5 | 51.3 -> 44.0 |

## Cross-dataset: gdino-tiny (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 831 | 65.2 -> 30.1 | 47.2 -> 9.1 | 71.6 -> 31.2 | 83.9 -> 45.6 |
| car | 5798 | 51.7 -> 51.0 | 44.1 -> 48.1 | 61.6 -> 50.1 | 75.8 -> 57.8 |
| truck | 219 | 28.0 -> 38.1 | 14.3 -> 49.5 | 29.8 -> 35.7 | 45.2 -> 40.3 |
| **all** |  | 49.9 -> 39.7 | 34.9 -> 35.6 | 52.9 -> 39.0 | 64.0 -> 47.9 |

## Cross-dataset: rtdetr-l (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 831 | 61.5 -> 28.3 | 40.3 -> 7.4 | 69.0 -> 29.1 | 83.0 -> 44.6 |
| car | 5798 | 49.9 -> 51.4 | 38.4 -> 44.0 | 66.1 -> 48.3 | 80.4 -> 60.6 |
| truck | 219 | 45.8 -> 17.8 | 23.0 -> 31.5 | 48.4 -> 17.8 | 62.8 -> 26.3 |
| **all** |  | 52.6 -> 32.5 | 33.9 -> 27.7 | 57.3 -> 31.7 | 70.7 -> 43.8 |

## Cross-dataset: yolo-world-s (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 831 | 56.3 -> 24.9 | 33.2 -> 2.1 | 65.1 -> 25.6 | 79.0 -> 45.5 |
| car | 5798 | 39.1 -> 48.1 | 27.6 -> 32.2 | 57.6 -> 45.6 | 63.8 -> 60.9 |
| truck | 219 | 29.1 -> 32.4 | 11.0 -> 46.3 | 29.2 -> 29.8 | 43.7 -> 44.1 |
| **all** |  | 36.6 -> 35.1 | 20.4 -> 26.9 | 40.5 -> 33.7 | 48.8 -> 50.2 |

## Cross-dataset: yolo11s (COCO -> kitti)

| class | #gt (kitti) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 831 | 58.6 -> 26.7 | 36.0 -> 5.2 | 67.3 -> 27.3 | 81.1 -> 43.1 |
| car | 5798 | 44.8 -> 47.0 | 32.2 -> 33.0 | 63.9 -> 44.6 | 73.9 -> 58.3 |
| truck | 219 | 35.5 -> 11.9 | 15.0 -> 9.1 | 35.7 -> 7.2 | 54.0 -> 23.5 |
| **all** |  | 45.5 -> 28.5 | 27.6 -> 15.8 | 50.1 -> 26.4 | 62.5 -> 41.6 |

## Cross-dataset: fasterrcnn_r50_fpn_v2 (COCO -> kitti-loose)

| class | #gt (kitti-loose) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 1197 | 59.6 -> 21.8 | 42.5 -> 10.4 | 66.2 -> 21.9 | 75.9 -> 30.5 |
| car | 6385 | 49.9 -> 48.0 | 40.7 -> 39.0 | 62.9 -> 45.8 | 69.9 -> 57.2 |
| truck | 219 | 42.7 -> 21.6 | 25.4 -> 10.2 | 44.5 -> 20.6 | 56.4 -> 32.2 |
| **all** |  | 46.9 -> 30.5 | 31.0 -> 19.9 | 50.8 -> 29.4 | 60.4 -> 39.9 |

## Cross-dataset: yolo11s (COCO -> kitti-loose)

| class | #gt (kitti-loose) | AP | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|
| person | 1197 | 58.6 -> 19.4 | 36.0 -> 5.6 | 67.3 -> 20.5 | 81.1 -> 29.2 |
| car | 6385 | 44.8 -> 45.3 | 32.2 -> 32.0 | 63.9 -> 42.8 | 73.9 -> 56.3 |
| truck | 219 | 35.5 -> 11.9 | 15.0 -> 9.1 | 35.7 -> 7.2 | 54.0 -> 23.5 |
| **all** |  | 45.5 -> 25.5 | 27.6 -> 15.6 | 50.1 -> 23.5 | 62.5 -> 36.3 |

## Speed / memory (batch 1, end-to-end)

| dataset | model | family | fp16 | params (M) | median ms | p90 ms | FPS | peak mem (MB) | AP |
|---|---|---|---|---|---|---|---|---|---|
| coco | fasterrcnn_r50_fpn_v2 | R-CNN | no | 43.71 | 86.9 | 99.5 | 11.5 | 672.3 | 46.9 |
| coco | fasterrcnn_r50_fpn_v2 | R-CNN | yes | 43.71 | 50.7 | 65.7 | 19.7 | 566.8 | 46.9 |
| coco | fcos_r50_fpn | RetinaNet/FCOS | no | 32.27 | 58.2 | 79.9 | 17.2 | 422.9 | 39.2 |
| coco | fcos_r50_fpn | RetinaNet/FCOS | yes | 32.27 | 63.2 | 102.9 | 15.8 | 389.3 | 39.2 |
| coco | gdino-tiny | Open-vocab | no | 172.25 | 1204.3 | 1337.7 | 0.8 | 6422.9 | 49.9 |
| coco | rtdetr-l | DETR | no | 32.97 | 50.0 | 62.4 | 20.0 | 238.6 | 52.6 |
| coco | rtdetr-l | DETR | yes | 32.97 | 39.6 | 47.5 | 25.2 | 143.2 | 52.6 |
| coco | yolo-world-s | Open-vocab | no | 12.76 | 17.1 | 22.4 | 58.5 | 129.4 | 36.6 |
| coco | yolo-world-s | Open-vocab | yes | 12.76 | 23.3 | 39.0 | 43.0 | 82.1 | 36.6 |
| coco | yolo11s | YOLO | no | 9.46 | 18.6 | 26.3 | 53.7 | 110.5 | 45.5 |
| coco | yolo11s | YOLO | yes | 9.46 | 22.4 | 35.0 | 44.6 | 73.1 | 45.5 |
| kitti | fasterrcnn_r50_fpn_v2 | R-CNN | no | 43.71 | 58.8 | 60.8 | 17.0 | 441.1 | 33.8 |
| kitti | fasterrcnn_r50_fpn_v2 | R-CNN | yes | 43.71 | 38.6 | 40.6 | 25.9 | 389.9 | 33.8 |
| kitti | fcos_r50_fpn | RetinaNet/FCOS | no | 32.27 | 37.3 | 39.2 | 26.8 | 283.3 | 33.5 |
| kitti | fcos_r50_fpn | RetinaNet/FCOS | yes | 32.27 | 41.0 | 47.2 | 24.4 | 266.2 | 33.5 |
| kitti | gdino-tiny | Open-vocab | no | 172.25 | 204.6 | 219.6 | 4.9 | 1367.3 | 39.7 |
| kitti | rtdetr-l | DETR | no | 32.97 | 43.2 | 49.0 | 23.1 | 238.6 | 32.5 |
| kitti | rtdetr-l | DETR | yes | 32.97 | 42.4 | 45.6 | 23.6 | 143.2 | 32.5 |
| kitti | yolo-world-s | Open-vocab | no | 12.76 | 19.3 | 22.1 | 51.9 | 99.0 | 35.1 |
| kitti | yolo-world-s | Open-vocab | yes | 12.76 | 19.6 | 22.4 | 51.0 | 65.9 | 35.1 |
| kitti | yolo11s | YOLO | no | 9.46 | 18.5 | 20.2 | 54.1 | 85.9 | 28.5 |
| kitti | yolo11s | YOLO | no | 9.46 | 19.7 | 21.7 | 50.8 | 123.0 | 32.5 |
| kitti | yolo11s | YOLO | yes | 9.46 | 20.1 | 23.0 | 49.9 | 58.7 | 28.5 |

## Configurations

| dataset | run | checkpoint | input | conf | NMS IoU | framework | command |
|---|---|---|---|---|---|---|---|
| coco | fasterrcnn_r50_fpn_v2 | FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.5 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fasterrcnn_r50_fpn_v2 --dataset coco` |
| coco | fcos_r50_fpn | FCOS_ResNet50_FPN_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.6 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fcos_r50_fpn --dataset coco` |
| coco | gdino-tiny | IDEA-Research/grounding-dino-tiny | processor default (short side 800, long side <= 1333) | 0.001 | none | transformers 5.21.0 | `python.exe evaluate.py --model gdino-tiny --dataset coco --max-images 1000` |
| coco | rtdetr-l | rtdetr-l.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | none (set prediction, NMS-free) | ultralytics 8.4.190 | `python.exe evaluate.py --model rtdetr-l --dataset coco` |
| coco | yolo-world-s | yolov8s-worldv2.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo-world-s --dataset coco` |
| coco | yolo11s | yolo11s.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo11s --dataset coco` |
| kitti | fasterrcnn_r50_fpn_v2 | FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.5 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fasterrcnn_r50_fpn_v2 --dataset kitti` |
| kitti | fcos_r50_fpn | FCOS_ResNet50_FPN_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.6 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fcos_r50_fpn --dataset kitti` |
| kitti | gdino-tiny | IDEA-Research/grounding-dino-tiny | processor default (short side 800, long side <= 1333) | 0.001 | none | transformers 5.21.0 | `python.exe evaluate.py --model gdino-tiny --dataset kitti` |
| kitti | rtdetr-l | rtdetr-l.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | none (set prediction, NMS-free) | ultralytics 8.4.190 | `python.exe evaluate.py --model rtdetr-l --dataset kitti` |
| kitti | rtdetr-l_1280 | rtdetr-l.pt | letterbox, long side 1280 (rect, stride-padded) | 0.001 | none (set prediction, NMS-free) | ultralytics 8.4.190 | `python.exe evaluate.py --model rtdetr-l --dataset kitti --imgsz 1280 --run-name rtdetr-l_1280` |
| kitti | yolo-world-s | yolov8s-worldv2.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo-world-s --dataset kitti` |
| kitti | yolo11s | yolo11s.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo11s --dataset kitti` |
| kitti | yolo11s_1280 | yolo11s.pt | letterbox, long side 1280 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo11s --dataset kitti --imgsz 1280 --run-name yolo11s_1280` |
| kitti-loose | fasterrcnn_r50_fpn_v2 | FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 | resize short side 800, long side <= 1333 | 0.001 | 0.5 | torchvision 0.26.0+cu128 | `python.exe evaluate.py --model fasterrcnn_r50_fpn_v2 --dataset kitti-loose` |
| kitti-loose | yolo11s | yolo11s.pt | letterbox, long side 640 (rect, stride-padded) | 0.001 | 0.7 | ultralytics 8.4.190 | `python.exe evaluate.py --model yolo11s --dataset kitti-loose` |
