"""Collect hardware + software versions for the reproducibility section of the report.

Usage:
    python env_info.py            # writes results/env.json and results/pip_freeze.txt
"""
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = ["torch", "torchvision", "ultralytics", "transformers", "pycocotools",
            "numpy", "pillow", "opencv-python", "timm"]

NVSMI_FIELDS = ["name", "driver_version", "memory.total", "power.max_limit",
                "enforced.power.limit", "clocks.max.graphics"]


def _cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        return None


def _nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(NVSMI_FIELDS)}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [dict(zip(NVSMI_FIELDS, (v.strip() for v in line.split(","))))
            for line in out.strip().splitlines()]


def collect_env():
    info = {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "ram_gb": _ram_gb(),
        "packages": {},
        "nvidia_smi": _nvidia_smi(),
    }
    for pkg in PACKAGES:
        try:
            info["packages"][pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpus"] = [{
            "name": torch.cuda.get_device_name(i),
            "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
        } for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    info = collect_env()
    (out_dir / "env.json").write_text(json.dumps(info, indent=2))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out_dir / "pip_freeze.txt").write_text(freeze)
    print(json.dumps(info, indent=2))
    print(f"\nWrote {out_dir / 'env.json'} and {out_dir / 'pip_freeze.txt'}")


if __name__ == "__main__":
    main()
