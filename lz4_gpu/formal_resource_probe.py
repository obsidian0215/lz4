#!/usr/bin/env python3
"""Collect process and machine-energy evidence for HeteroLZ commands."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    import resource
except ImportError:  # Formal runs are Linux-only; keep the module importable elsewhere.
    resource = None


SCHEMA = "heterolz.process-resource.v1"


@dataclass(frozen=True)
class EnergyDomain:
    name: str
    energy_path: Path
    max_energy_uj: float


class EnergyProbe:
    """Read machine-level energy sources without pretending they are process-local."""

    def __init__(self, powercap_root: Path = Path("/sys/class/powercap"),
                 nvidia_smi: Path | None = None) -> None:
        self._helper_user_s = 0.0
        self._helper_system_s = 0.0
        self.powercap_root = powercap_root
        self.cpu_package = self._find_cpu_package()
        discovered = shutil.which("nvidia-smi") if nvidia_smi is None else str(nvidia_smi)
        self.nvidia_smi = Path(discovered).resolve() if discovered else None
        self.nvidia_energy_supported = self._read_nvidia("total_energy_consumption") is not None
        self.nvidia_power_supported = (
            not self.nvidia_energy_supported and self._read_nvidia("power.draw") is not None
        )

    @staticmethod
    def _read_float(path: Path) -> float | None:
        try:
            value = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _find_cpu_package(self) -> EnergyDomain | None:
        if not self.powercap_root.is_dir():
            return None
        requested_name: str | None = None
        cpu_set = os.environ.get("HETEROLZ_CPU_SET", "").split(",", 1)[0].strip()
        if cpu_set.isdigit():
            try:
                package_id = (Path("/sys/devices/system/cpu") / f"cpu{cpu_set}" /
                              "topology" / "physical_package_id").read_text(
                                  encoding="utf-8"
                              ).strip()
            except OSError:
                package_id = ""
            if package_id.lstrip("-").isdigit():
                requested_name = f"package-{package_id}"
        candidates: list[EnergyDomain] = []
        for directory in sorted(self.powercap_root.rglob("intel-rapl*")):
            if not directory.is_dir():
                continue
            try:
                name = (directory / "name").read_text(encoding="utf-8").strip()
            except OSError:
                continue
            energy_path = directory / "energy_uj"
            energy = self._read_float(energy_path)
            if not name.startswith("package") or energy is None:
                continue
            maximum = self._read_float(directory / "max_energy_range_uj") or 0.0
            candidates.append(EnergyDomain(name, energy_path, maximum))
        if requested_name is not None:
            return next((item for item in candidates if item.name == requested_name), None)
        return candidates[0] if candidates else None

    def _read_nvidia(self, field: str) -> float | None:
        if self.nvidia_smi is None or not self.nvidia_smi.is_file():
            return None
        usage_start = _child_usage()
        completed = subprocess.run(
            [str(self.nvidia_smi), f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        usage_end = _child_usage()
        self._helper_user_s += max(0.0, usage_end[0] - usage_start[0])
        self._helper_system_s += max(0.0, usage_end[1] - usage_start[1])
        if completed.returncode != 0:
            return None
        lines = completed.stdout.strip().splitlines()
        if not lines:
            return None
        try:
            values = [float(line.split(",", 1)[0].strip()) for line in lines]
        except ValueError:
            return None
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            return None
        return sum(values)

    def reset_helper_usage(self) -> None:
        self._helper_user_s = 0.0
        self._helper_system_s = 0.0

    def helper_usage(self) -> tuple[float, float]:
        return self._helper_user_s, self._helper_system_s

    def snapshot(self) -> dict[str, int | float | None]:
        cpu_energy = self._read_float(self.cpu_package.energy_path) if self.cpu_package else None
        gpu_energy_mj = (
            self._read_nvidia("total_energy_consumption")
            if self.nvidia_energy_supported else None
        )
        gpu_power_w = self._read_nvidia("power.draw") if self.nvidia_power_supported else None
        return {
            "time_ns": time.perf_counter_ns(),
            "cpu_package_energy_uj": cpu_energy,
            "gpu_energy_mj": gpu_energy_mj,
            "gpu_power_w": gpu_power_w,
        }

    @staticmethod
    def _delta_counter(start: float | None, end: float | None,
                       maximum: float) -> float | None:
        if start is None or end is None:
            return None
        if end >= start:
            return end - start
        if maximum > 0.0:
            return (maximum - start) + end
        return None

    @classmethod
    def _sum_counter_deltas(cls, samples: Sequence[Mapping[str, int | float | None]],
                            field: str, maximum: float) -> float | None:
        if len(samples) < 2:
            return None
        total = 0.0
        for left, right in zip(samples, samples[1:]):
            delta = cls._delta_counter(
                _as_float(left.get(field)), _as_float(right.get(field)), maximum
            )
            if delta is None:
                return None
            total += delta
        return total

    def summarize(self, samples: Sequence[Mapping[str, int | float | None]]) -> dict[str, object]:
        cpu_energy_j: float | None = None
        gpu_energy_j: float | None = None
        if len(samples) >= 2:
            cpu_uj = self._sum_counter_deltas(
                samples,
                "cpu_package_energy_uj",
                self.cpu_package.max_energy_uj if self.cpu_package else 0.0,
            )
            if cpu_uj is not None:
                cpu_energy_j = cpu_uj * 1e-6

            gpu_mj = self._sum_counter_deltas(samples, "gpu_energy_mj", 0.0)
            if gpu_mj is not None:
                gpu_energy_j = gpu_mj * 1e-3
            elif self.nvidia_power_supported:
                integrated = 0.0
                complete = True
                for left, right in zip(samples, samples[1:]):
                    p0 = _as_float(left.get("gpu_power_w"))
                    p1 = _as_float(right.get("gpu_power_w"))
                    t0 = _as_float(left.get("time_ns"))
                    t1 = _as_float(right.get("time_ns"))
                    if p0 is None or p1 is None or t0 is None or t1 is None or t1 <= t0:
                        complete = False
                        break
                    integrated += (p0 + p1) * 0.5 * ((t1 - t0) / 1_000_000_000.0)
                if complete:
                    gpu_energy_j = integrated

        return {
            "scope": "process_wall_with_machine_energy",
            "cpu_package_energy_j": cpu_energy_j,
            "gpu_energy_j": gpu_energy_j,
            "sources": {
                "cpu_package": (
                    str(self.cpu_package.energy_path.resolve())
                    if self.cpu_package else "unavailable"
                ),
                "gpu": (
                    f"{self.nvidia_smi}:all-gpus:total_energy_consumption"
                    if self.nvidia_energy_supported and self.nvidia_smi else
                    f"{self.nvidia_smi}:all-gpus:power.draw"
                    if self.nvidia_power_supported and self.nvidia_smi else "unavailable"
                ),
            },
            "counter_metadata": {
                "cpu_package_max_energy_uj": (
                    self.cpu_package.max_energy_uj if self.cpu_package else None
                ),
            },
            "energy_samples": [dict(sample) for sample in samples],
        }


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _process_peak_rss_bytes(pid: int) -> int:
    try:
        lines = (Path("/proc") / str(pid) / "status").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return 0
    values: dict[str, int] = {}
    for line in lines:
        if not line.startswith(("VmHWM:", "VmRSS:")):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            values[fields[0].rstrip(":")] = int(fields[1]) * 1024
    return max(values.get("VmHWM", 0), values.get("VmRSS", 0))


def _child_usage() -> tuple[float, float]:
    if resource is None:
        return (0.0, 0.0)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (float(usage.ru_utime), float(usage.ru_stime))


def run_measured(argv: Sequence[str], *, cwd: Path | None = None,
                 env: Mapping[str, str] | None = None,
                 interval_s: float = 0.02,
                 energy_probe: EnergyProbe | None = None,
                 timeout_s: float | None = None) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    if not argv:
        raise ValueError("argv must not be empty")
    if interval_s <= 0.0:
        raise ValueError("interval_s must be positive")
    probe = energy_probe or EnergyProbe()
    first_energy_sample = probe.snapshot()
    probe.reset_helper_usage()
    usage_start = _child_usage()
    wall_start = time.perf_counter_ns()
    process = subprocess.Popen(
        list(argv), cwd=cwd, env=dict(env) if env is not None else None,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    peak_rss_bytes = _process_peak_rss_bytes(process.pid)
    samples = [first_energy_sample]
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(list(argv), timeout_s, output=stdout, stderr=stderr)
        wait_s = interval_s
        if deadline is not None:
            wait_s = min(wait_s, max(0.001, deadline - time.monotonic()))
        try:
            stdout, stderr = process.communicate(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            peak_rss_bytes = max(peak_rss_bytes, _process_peak_rss_bytes(process.pid))
            samples.append(probe.snapshot())
    wall_end = time.perf_counter_ns()
    samples.append(probe.snapshot())
    usage_end = _child_usage()
    helper_user_s, helper_system_s = probe.helper_usage()
    energy = probe.summarize(samples)
    completed = subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)
    measurement = {
        "schema": SCHEMA,
        "status": "complete",
        "process_returncode": process.returncode,
        "process_elapsed_us": (wall_end - wall_start) / 1000.0,
        "cpu_user_us": max(0.0, usage_end[0] - usage_start[0] - helper_user_s) * 1_000_000.0,
        "cpu_system_us": max(0.0, usage_end[1] - usage_start[1] - helper_system_s) * 1_000_000.0,
        "process_cpu_time_source": "rusage_children_minus_probe_helpers",
        "peak_rss_bytes": peak_rss_bytes,
        **energy,
    }
    return completed, measurement
