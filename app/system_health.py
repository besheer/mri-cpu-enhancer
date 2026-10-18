from __future__ import annotations

import os
import time

import httpx
import psutil


def snapshot() -> dict[str, object]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/var/lib/enhancer")
    process = psutil.Process(os.getpid())
    orthanc_ok = False
    try:
        response = httpx.get(
            os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/") + "/system",
            auth=(os.getenv("ORTHANC_USERNAME", "admin"), os.getenv("ORTHANC_PASSWORD", "")),
            timeout=3,
        )
        orthanc_ok = response.status_code == 200
    except httpx.HTTPError:
        pass
    return {
        "timestamp": int(time.time()),
        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "cpu_count": psutil.cpu_count() or 1,
        "ram_percent": round(memory.percent, 1),
        "ram_used_gb": round(memory.used / 1024**3, 2),
        "ram_total_gb": round(memory.total / 1024**3, 2),
        "disk_percent": round(disk.percent, 1),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "process_ram_mb": round(process.memory_info().rss / 1024**2, 1),
        "orthanc": "UP" if orthanc_ok else "DOWN",
    }
