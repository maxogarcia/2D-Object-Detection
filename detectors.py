"""Uniform wrappers around detectors from different frameworks / families.

Every detector exposes
    predict(img_rgb: HxWx3 uint8 ndarray) -> (boxes_xyxy [N,4], scores [N], coco_category_ids [N])
in original-image pixel coordinates, a `config` dict (written into every results file), and
`module` (the underlying torch.nn.Module, used for parameter counts).
"""
import numpy as np
import torch

# Ultralytics / most YOLO repos use contiguous 0..79 indices; COCO JSON uses ids 1..90 with gaps.
COCO80_TO_COCO91 = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90])

# name -> (backend, family, default weights)
REGISTRY = {
    # YOLO family (one-stage, anchor-free, NMS)
    "yolo11n": ("ultralytics-yolo", "YOLO", "yolo11n.pt"),
    "yolo11s": ("ultralytics-yolo", "YOLO", "yolo11s.pt"),
    "yolo11m": ("ultralytics-yolo", "YOLO", "yolo11m.pt"),
    "yolo11l": ("ultralytics-yolo", "YOLO", "yolo11l.pt"),
    "yolov8s": ("ultralytics-yolo", "YOLO", "yolov8s.pt"),
    # DETR family (transformer decoder, set prediction, no NMS)
    "rtdetr-l": ("ultralytics-rtdetr", "DETR", "rtdetr-l.pt"),
    "rtdetr-x": ("ultralytics-rtdetr", "DETR", "rtdetr-x.pt"),
    # Two-stage R-CNN family
    "fasterrcnn_r50_fpn_v2": ("torchvision", "R-CNN", "fasterrcnn_resnet50_fpn_v2"),
    # Dense one-stage (anchor-based / anchor-free) family
    "retinanet_r50_fpn_v2": ("torchvision", "RetinaNet/FCOS", "retinanet_resnet50_fpn_v2"),
    "fcos_r50_fpn": ("torchvision", "RetinaNet/FCOS", "fcos_resnet50_fpn"),
    # Open-vocabulary family
    "yolo-world-s": ("ultralytics-world", "Open-vocab", "yolov8s-worldv2.pt"),
    "yolo-world-l": ("ultralytics-world", "Open-vocab", "yolov8l-worldv2.pt"),
    "gdino-tiny": ("hf-gdino", "Open-vocab", "IDEA-Research/grounding-dino-tiny"),
    "gdino-base": ("hf-gdino", "Open-vocab", "IDEA-Research/grounding-dino-base"),
}

TORCHVISION_MODELS = {
    # builder name: (weights enum name, kwarg prefix for thresholds)
    "fasterrcnn_resnet50_fpn_v2": ("FasterRCNN_ResNet50_FPN_V2_Weights", "box_"),
    "retinanet_resnet50_fpn_v2": ("RetinaNet_ResNet50_FPN_V2_Weights", ""),
    "fcos_resnet50_fpn": ("FCOS_ResNet50_FPN_Weights", ""),
}


class UltralyticsDetector:
    def __init__(self, backend, weights, device, conf, iou, imgsz, max_det, half, categories):
        import ultralytics
        from ultralytics import RTDETR, YOLO, YOLOWorld
        cls = {"ultralytics-yolo": YOLO, "ultralytics-rtdetr": RTDETR, "ultralytics-world": YOLOWorld}[backend]
        self.model = cls(weights)
        if backend == "ultralytics-world":
            if not categories:
                raise ValueError("Open-vocabulary detector needs the dataset's category list")
            self.model.set_classes([c["name"] for c in categories])
            self.idx_to_cat = np.array([c["id"] for c in categories])
            prompt = [c["name"] for c in categories]
        else:
            self.idx_to_cat = COCO80_TO_COCO91
            prompt = None
        iou = 0.7 if iou is None else iou  # Ultralytics' own val default
        imgsz = 640 if imgsz is None else imgsz
        self.kw = dict(conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, half=half,
                       device=device, verbose=False)
        self.config = {
            "framework": f"ultralytics {ultralytics.__version__}", "checkpoint": weights,
            "input_resolution": f"letterbox, long side {imgsz} (rect, stride-padded)",
            "conf_thresh": conf,
            "nms_iou": "none (set prediction, NMS-free)" if backend == "ultralytics-rtdetr" else iou,
            "max_det": max_det, "fp16": half, "text_prompt": prompt,
        }

    @property
    def module(self):
        return self.model.model

    def predict(self, img_rgb):
        r = self.model.predict(np.ascontiguousarray(img_rgb[..., ::-1]), **self.kw)[0]  # expects BGR
        b = r.boxes
        cls = b.cls.cpu().numpy().astype(int)
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), self.idx_to_cat[cls]


class TorchvisionDetector:
    def __init__(self, builder, device, conf, iou, max_det, half, min_size, max_size):
        import torchvision
        from torchvision.models import detection
        weights_enum, prefix = TORCHVISION_MODELS[builder]
        weights = getattr(detection, weights_enum).DEFAULT
        kw = {f"{prefix}score_thresh": conf, f"{prefix}detections_per_img": max_det}
        if iou is not None:
            kw[f"{prefix}nms_thresh"] = iou
        if min_size is not None:
            kw["min_size"] = min_size
        if max_size is not None:
            kw["max_size"] = max_size
        self.model = getattr(detection, builder)(weights=weights, **kw).to(device).eval()
        self.device, self.half = device, half
        nms = self.model.roi_heads.nms_thresh if hasattr(self.model, "roi_heads") else self.model.nms_thresh
        t = self.model.transform
        self.config = {
            "framework": f"torchvision {torchvision.__version__}", "checkpoint": str(weights),
            "input_resolution": f"resize short side {t.min_size[0]}, long side <= {t.max_size}",
            "conf_thresh": conf, "nms_iou": nms, "max_det": max_det, "fp16": half,
        }

    @property
    def module(self):
        return self.model

    @torch.inference_mode()
    def predict(self, img_rgb):
        x = torch.from_numpy(img_rgb).to(self.device).permute(2, 0, 1).float().div_(255)
        with torch.autocast("cuda", enabled=self.half):
            out = self.model([x])[0]
        # torchvision COCO weights already use the 91-id COCO category space
        return (out["boxes"].float().cpu().numpy(), out["scores"].float().cpu().numpy(),
                out["labels"].cpu().numpy())


class GroundingDinoDetector:
    """Grounding DINO via Hugging Face transformers.

    The text encoder has a 256-token limit, so the class list is split into chunks and the image is
    run once per chunk (COCO's 80 classes -> 4 passes at chunk=20). Results are merged and the top
    `max_det` by score are kept. Scoring is done here (vectorised) rather than with HF's
    post_process_grounded_object_detection, which decodes a text label per box and is very slow at
    the low thresholds mAP evaluation needs.
    """

    def __init__(self, weights, device, conf, max_det, half, categories, chunk=20):
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        if not categories:
            raise ValueError("Open-vocabulary detector needs the dataset's category list")
        self.processor = AutoProcessor.from_pretrained(weights)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(weights).to(device).eval()
        self.device, self.half, self.conf, self.max_det = device, half, conf, max_det
        self.chunks = [categories[i:i + chunk] for i in range(0, len(categories), chunk)]
        self.prompts = [". ".join(c["name"] for c in ch) + "." for ch in self.chunks]
        self.positive_maps = [self._positive_map(p, ch) for p, ch in zip(self.prompts, self.chunks)]
        self.config = {
            "framework": f"transformers {transformers.__version__}", "checkpoint": weights,
            "input_resolution": "processor default (short side 800, long side <= 1333)",
            "conf_thresh": conf, "nms_iou": "none", "max_det": max_det, "fp16": half,
            "scoring": "class score = mean sigmoid over the class name's tokens; top-k over queries x classes "
                       "(as in the official Grounding DINO COCO evaluation)",
            "text_prompts": self.prompts,
        }

    def _positive_map(self, prompt, chunk):
        """[num_classes, num_tokens] matrix, row-normalised over the tokens of each class name."""
        offsets = self.processor.tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
        pmap = torch.zeros(len(chunk), len(offsets))
        pos = 0
        for k, c in enumerate(chunk):
            a = prompt.index(c["name"], pos)
            b = pos = a + len(c["name"])
            toks = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
            pmap[k, toks] = 1.0 / len(toks)
        return pmap.to(self.device)

    @property
    def module(self):
        return self.model

    @torch.inference_mode()
    def predict(self, img_rgb):
        from PIL import Image
        img = Image.fromarray(img_rgb)
        h, w = img_rgb.shape[:2]
        scale = torch.tensor([w, h, w, h], device=self.device)
        # all prompt chunks go through the model as one batch (same image repeated); padding is on the
        # right, so each chunk's token positions (and positive map) are unchanged
        inputs = self.processor(images=[img] * len(self.prompts), text=self.prompts, padding="longest",
                                return_tensors="pt").to(self.device)
        with torch.autocast("cuda", enabled=self.half):
            outputs = self.model(**inputs)
        boxes, scores, cats = [], [], []
        for b, (chunk, pmap) in enumerate(zip(self.chunks, self.positive_maps)):
            prob = outputs.logits[b].float().sigmoid()[:, :pmap.shape[1]]  # [queries, tokens]
            cls_scores = prob @ pmap.T                                       # [queries, classes]
            top = cls_scores.flatten().topk(min(self.max_det, cls_scores.numel()))
            q, k = top.indices // cls_scores.shape[1], top.indices % cls_scores.shape[1]
            keep = top.values >= self.conf
            cx, cy, bw, bh = outputs.pred_boxes[b].float()[q[keep]].unbind(-1)
            boxes.append((torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * scale).cpu().numpy())
            scores.append(top.values[keep].cpu().numpy())
            cats.append(np.array([chunk[i]["id"] for i in k[keep].tolist()], dtype=int))
        boxes, scores, cats = np.concatenate(boxes), np.concatenate(scores), np.concatenate(cats)
        keep = np.argsort(-scores)[:self.max_det]
        return boxes[keep], scores[keep], cats[keep]


def build_detector(name, device="cuda:0", conf=0.001, iou=None, imgsz=None, max_det=100, half=False,
                   categories=None, weights=None, tv_min_size=None, tv_max_size=None):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Choices: {', '.join(REGISTRY)}")
    backend, family, default_weights = REGISTRY[name]
    weights = weights or default_weights
    if backend.startswith("ultralytics"):
        det = UltralyticsDetector(backend, weights, device, conf, iou, imgsz, max_det, half, categories)
    elif backend == "torchvision":
        det = TorchvisionDetector(weights, device, conf, iou, max_det, half, tv_min_size, tv_max_size)
    elif backend == "hf-gdino":
        det = GroundingDinoDetector(weights, device, conf, max_det, half, categories)
    else:
        raise ValueError(backend)
    det.config = {"model": name, "family": family, **det.config,
                  "params_M": round(sum(p.numel() for p in det.module.parameters()) / 1e6, 2)}
    return det
