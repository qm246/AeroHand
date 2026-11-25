"""Lightweight cached System Vitals monitor (Airpoint `vitals` parity)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


@dataclass
class SystemVitals:
    cpu_usage: float = 0.0
    ram_total_mb: float = 0.0
    ram_used_mb: float = 0.0
    ram_percent: float = 0.0
    process_ram_mb: float = 0.0
    temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "cpu_usage": round(self.cpu_usage, 1),
            "ram_total": round(self.ram_total_mb * 1024.0, 0),  # KB for Airpoint CLI compat
            "ram_used": round(self.ram_used_mb * 1024.0, 0),
            "ram_total_gb": round(self.ram_total_mb / 1024.0, 2),
            "ram_used_gb": round(self.ram_used_mb / 1024.0, 2),
            "ram_percent": round(self.ram_percent, 1),
            "process_ram_mb": round(self.process_ram_mb, 1),
            "temperature": self.temperature,
        }


class SystemVitalsMonitor:
    """Samples macOS CPU/RAM lazily with a minimum interval to keep overhead near zero."""

    def __init__(self, refresh_interval_s: float = 1.5) -> None:
        self._interval = max(0.5, float(refresh_interval_s))
        self._last_t: float = 0.0
        self._cached = SystemVitals()
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def read(self, force: bool = False) -> SystemVitals:
        now = time.monotonic()
        if not force and (now - self._last_t) < self._interval:
            return self._cached
        self._last_t = now
        if psutil is None:
            return self._cached
        try:
            cpu = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            rss_mb = float(self._proc.memory_info().rss) / (1024.0 * 1024.0) if self._proc else 0.0
            self._cached = SystemVitals(
                cpu_usage=cpu,
                ram_total_mb=float(vm.total) / (1024.0 * 1024.0),
                ram_used_mb=float(vm.used) / (1024.0 * 1024.0),
                ram_percent=float(vm.percent),
                process_ram_mb=rss_mb,
                temperature=None,
            )
        except Exception:
            pass
        return self._cached
