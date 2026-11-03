"""Collect every results/*/*/metrics.json and results/bench/*.json into report-ready tables.

Writes results/summary.md (Markdown tables you can paste into the report) and results/summary.csv.

Usage:
    python summarize.py [--results results]
"""
import argparse
import csv
import json
from pathlib import Path


def fmt(v, scale=100, nd=1):
    return "-" if v is None else f"{v * scale:.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    RESULTS = Path(ap.parse_args().results)
    runs = []
    for p in sorted(RESULTS.glob("*/*/metrics.json")):
        m = json.loads(p.read_text())
        runs.append({**m, "ds": p.parent.parent.name, "run": p.parent.name})
    benches = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("bench/*.json"))]

    md = ["# Results summary", "", "AP values are COCO-style, x100.", ""]

    # 1. Main accuracy table
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR100"]
    rows, csv_rows = [], []
    for r in runs:
        s = r["stats"]
        rows.append([r["ds"], r["run"], r["model"]["family"], r["dataset"]["num_images"], *[fmt(s[k]) for k in keys]])
        csv_rows.append({"dataset": r["ds"], "run": r["run"], "family": r["model"]["family"], **s})
    md += ["## Accuracy", "", table(["dataset", "run", "family", "#img", *keys], rows), ""]

    # 2. Per-class AP per dataset
    for ds in sorted({r["ds"] for r in runs}):
        ds_runs = [r for r in runs if r["ds"] == ds]
        classes = [c for c in ds_runs[0]["per_class"] if c["num_gt"]]
        if len(classes) > 15:  # COCO: keep the table readable; full numbers are in metrics.json
            classes = sorted(classes, key=lambda c: -c["num_gt"])[:15]
        names = [c["name"] for c in classes]
        rows = []
        for r in ds_runs:
            by_name = {c["name"]: c for c in r["per_class"]}
            rows.append([r["run"], *[fmt(by_name.get(n, {}).get("AP")) for n in names]])
        title = f"## Per-class AP - {ds}" + (" (15 most frequent classes)" if len(ds_runs[0]["per_class"]) > 15 else "")
        md += [title, "", table(["run", *names], rows), ""]

    # 3. Cross-dataset degradation: same model on COCO vs each KITTI variant
    coco = {r["run"]: r for r in runs if r["ds"] == "coco"}
    for r in runs:
        if r["ds"] == "coco" or r["run"] not in coco:
            continue
        c = coco[r["run"]]
        c_cls = {x["category_id"]: x for x in c["per_class"]}
        rows = []
        for x in r["per_class"]:
            if not x["num_gt"]:
                continue
            y = c_cls.get(x["category_id"], {})
            rows.append([x["name"], x["num_gt"], *[f"{fmt(y.get(k))} -> {fmt(x.get(k))}"
                                                  for k in ("AP", "AP_small", "AP_medium", "AP_large")]])
        s0, s1 = c["stats"], r["stats"]
        rows.append(["**all**", "", *[f"{fmt(s0[k])} -> {fmt(s1[k])}" for k in ("AP", "AP_small", "AP_medium", "AP_large")]])
        md += [f"## Cross-dataset: {r['run']} (COCO -> {r['ds']})", "",
               table(["class", f"#gt ({r['ds']})", "AP", "AP_s", "AP_m", "AP_l"], rows), ""]

    # 4. Speed / memory
    if benches:
        acc = {(r["ds"], r["run"]): r["stats"]["AP"] for r in runs}
        rows = []
        for b in benches:
            name = b["model"]["model"]
            rows.append([b["dataset"], name, b["model"]["family"], "yes" if b["model"]["fp16"] else "no",
                         b["model"]["params_M"], f"{b['latency_ms']['median']:.1f}", f"{b['latency_ms']['p90']:.1f}",
                         f"{b['fps_from_median']:.1f}", b["peak_gpu_mem_allocated_MB"],
                         fmt(acc.get((b["dataset"], b.get("eval_run", name))))])
        md += ["## Speed / memory (batch 1, end-to-end)", "",
               table(["dataset", "model", "family", "fp16", "params (M)", "median ms", "p90 ms", "FPS",
                      "peak mem (MB)", "AP"], rows), ""]

    # 5. Reproducibility table
    rows = [[r["ds"], r["run"], r["model"]["checkpoint"], r["model"]["input_resolution"],
             r["model"].get("conf_thresh", r["model"].get("conf_thresh (box_threshold)")), r["model"]["nms_iou"],
             r["model"]["framework"], f"`{r['command']}`"] for r in runs]
    md += ["## Configurations", "",
           table(["dataset", "run", "checkpoint", "input", "conf", "NMS IoU", "framework", "command"], rows), ""]

    (RESULTS / "summary.md").write_text("\n".join(md), encoding="utf-8")
    if csv_rows:
        with open(RESULTS / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    print("\n".join(md))
    print(f"\nWrote {RESULTS / 'summary.md'} and {RESULTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
