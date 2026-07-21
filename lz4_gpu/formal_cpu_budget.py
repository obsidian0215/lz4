#!/usr/bin/env python3
"""CPU resource budget helpers for HeteroLZ formal runs."""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from pathlib import Path


FORMAL_CPU_THREADS = 4
PERFORMANCE_MEAN_BUSY_LIMIT_PCT = 15.0
PERFORMANCE_PEAK_BUSY_LIMIT_PCT = 35.0


def parse_cpu_set(value: str) -> tuple[int, ...]:
    items: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            raise ValueError("CPU set contains an empty item")
        if "-" in token:
            fields = token.split("-", 1)
            if len(fields) != 2 or not fields[0].isdigit() or not fields[1].isdigit():
                raise ValueError(f"invalid CPU range: {token}")
            start, end = (int(field) for field in fields)
            if end < start:
                raise ValueError(f"descending CPU range: {token}")
            items.extend(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"invalid CPU id: {token}")
            items.append(int(token))
    if not items or len(set(items)) != len(items):
        raise ValueError("CPU set must contain unique CPU ids")
    return tuple(items)


def _read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return default


def _read_optional_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_optional_text(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _cpu_online(cpu_dir: Path) -> bool:
    online = cpu_dir / "online"
    return not online.exists() or _read_int(online, 0) == 1


def _cpu_class(cpu_dir: Path) -> tuple[str, int | None, str]:
    core_type = _read_optional_int(cpu_dir / "topology" / "core_type")
    if core_type is not None:
        return "core_type", core_type, f"core_type:{core_type}"
    capacity = _read_optional_int(cpu_dir / "cpu_capacity")
    if capacity is not None:
        return "cpu_capacity", capacity, f"cpu_capacity:{capacity}"
    return "uniform", None, "uniform"


def _cpu_record(cpu_dir: Path) -> dict[str, object]:
    cpu = int(cpu_dir.name[3:])
    topology = cpu_dir / "topology"
    class_source, class_id, core_class = _cpu_class(cpu_dir)
    return {
        "cpu": cpu,
        "package": _read_int(topology / "physical_package_id", -1),
        "core": _read_int(topology / "core_id", -1),
        "core_class_source": class_source,
        "core_class_id": class_id,
        "core_class": core_class,
    }


def discover_cpu_set(count: int = FORMAL_CPU_THREADS,
                     sys_cpu_root: Path = Path("/sys/devices/system/cpu"),
                     allowed_cpus: tuple[int, ...] | None = None) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("CPU count must be positive")
    allowed = set(allowed_cpus) if allowed_cpus is not None else None
    records: list[dict[str, object]] = []
    for cpu_dir in sys_cpu_root.glob("cpu[0-9]*"):
        suffix = cpu_dir.name[3:]
        if not suffix.isdigit() or not _cpu_online(cpu_dir):
            continue
        if allowed is not None and int(suffix) not in allowed:
            continue
        record = _cpu_record(cpu_dir)
        if int(record["package"]) < 0 or int(record["core"]) < 0:
            continue
        records.append(record)
    records.sort(key=lambda item: int(item["cpu"]))
    if len(records) < count:
        raise RuntimeError(f"host exposes only {len(records)} online CPUs; {count} are required")

    groups: dict[tuple[int, str, int | None], dict[tuple[int, int], int]] = defaultdict(dict)
    for record in records:
        group = (int(record["package"]), str(record["core_class_source"]),
                 record["core_class_id"] if isinstance(record["core_class_id"], int) else None)
        physical_core = (int(record["package"]), int(record["core"]))
        groups[group].setdefault(physical_core, int(record["cpu"]))

    source_rank = {"uniform": 0, "cpu_capacity": 1, "core_type": 2}
    candidates = [(group, cores) for group, cores in groups.items() if len(cores) >= count]
    if not candidates:
        raise RuntimeError(
            f"host exposes fewer than {count} same-package, same-class physical cores"
        )
    group, cores = max(
        candidates,
        key=lambda item: (
            source_rank[item[0][1]],
            item[0][2] if item[0][2] is not None else -1,
            len(item[1]),
            -item[0][0],
        ),
    )
    del group
    return tuple(sorted(cores.values())[:count])


def collect_cpu_topology(cpus: tuple[int, ...],
                         sys_cpu_root: Path = Path("/sys/devices/system/cpu")) -> list[dict[str, object]]:
    topology: list[dict[str, object]] = []
    physical_cores: set[tuple[int, int]] = set()
    packages: set[int] = set()
    core_classes: set[tuple[str, int | None]] = set()
    for cpu in cpus:
        cpu_dir = sys_cpu_root / f"cpu{cpu}"
        if not cpu_dir.is_dir() or not _cpu_online(cpu_dir):
            raise RuntimeError(f"CPU {cpu} is missing or offline")
        record = _cpu_record(cpu_dir)
        package = int(record["package"])
        core = int(record["core"])
        if package < 0 or core < 0:
            raise RuntimeError(f"CPU {cpu} has incomplete topology data")
        key = (package, core)
        if key in physical_cores:
            raise RuntimeError("formal CPU set must use distinct physical cores")
        physical_cores.add(key)
        packages.add(package)
        class_id = record["core_class_id"] if isinstance(record["core_class_id"], int) else None
        core_classes.add((str(record["core_class_source"]), class_id))
        cpufreq = cpu_dir / "cpufreq"
        record.update(
            {
                "scaling_cur_freq_khz": _read_optional_int(cpufreq / "scaling_cur_freq"),
                "scaling_min_freq_khz": _read_optional_int(cpufreq / "scaling_min_freq"),
                "scaling_max_freq_khz": _read_optional_int(cpufreq / "scaling_max_freq"),
                "scaling_governor": _read_optional_text(cpufreq / "scaling_governor"),
            }
        )
        topology.append(record)
    if len(packages) != 1:
        raise RuntimeError("formal CPU set must use one physical package")
    if len(core_classes) != 1:
        raise RuntimeError("formal CPU set must use one CPU core class")
    return topology


def apply_process_affinity(cpus: tuple[int, ...]) -> None:
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("formal CPU affinity requires Linux sched_setaffinity")
    os.sched_setaffinity(0, set(cpus))
    applied = tuple(sorted(os.sched_getaffinity(0)))
    if applied != tuple(sorted(cpus)):
        raise RuntimeError(f"CPU affinity mismatch: requested={cpus} applied={applied}")


def apply_thread_environment(cpu_threads: int = FORMAL_CPU_THREADS) -> None:
    if cpu_threads < 1:
        raise ValueError("CPU thread budget must be positive")
    value = str(cpu_threads)
    for name in (
        "HETEROLZ_CPU_THREADS",
        "LZ4_NBWORKERS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value


def format_cpu_set(cpus: tuple[int, ...]) -> str:
    if not cpus:
        raise ValueError("CPU set must not be empty")
    return ",".join(str(cpu) for cpu in cpus)


def apply_cpu_environment(cpu_set: tuple[int, ...],
                          discovery_set: tuple[int, ...]) -> None:
    if not set(cpu_set).issubset(discovery_set):
        raise ValueError("formal CPU set must be contained in the discovery affinity")
    os.environ["HETEROLZ_CPU_SET"] = format_cpu_set(cpu_set)
    os.environ["HETEROLZ_CPU_DISCOVERY_SET"] = format_cpu_set(discovery_set)


def _read_proc_stat(path: Path = Path("/proc/stat")) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
            continue
        values = [int(value) for value in fields[1:]]
        if len(values) < 4:
            continue
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        result[int(fields[0][3:])] = (total, idle)
    return result


def sample_cpu_busy(cpus: tuple[int, ...], interval_seconds: float = 0.25) -> dict[int, float]:
    if interval_seconds <= 0:
        raise ValueError("CPU utilization interval must be positive")
    before = _read_proc_stat()
    time.sleep(interval_seconds)
    after = _read_proc_stat()
    result: dict[int, float] = {}
    for cpu in cpus:
        if cpu not in before or cpu not in after:
            raise RuntimeError(f"CPU {cpu} is missing from /proc/stat")
        total_delta = after[cpu][0] - before[cpu][0]
        idle_delta = after[cpu][1] - before[cpu][1]
        if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
            raise RuntimeError(f"invalid utilization counters for CPU {cpu}")
        result[cpu] = 100.0 * (total_delta - idle_delta) / total_delta
    return result


def collect_resource_evidence(cpus: tuple[int, ...], cpu_threads: int,
                              interval_seconds: float = 0.25) -> dict[str, object]:
    topology = collect_cpu_topology(cpus)
    busy = sample_cpu_busy(cpus, interval_seconds)
    values = list(busy.values())
    load_one = os.getloadavg()[0]
    if not math.isfinite(load_one) or load_one < 0:
        raise RuntimeError("invalid one-minute load average")
    return {
        "cpu_threads": cpu_threads,
        "cpu_set": list(cpus),
        "cpu_topology": topology,
        "physical_core_count": len(topology),
        "affinity": sorted(os.sched_getaffinity(0)),
        "busy_sample_seconds": interval_seconds,
        "busy_pct_by_cpu": {str(cpu): busy[cpu] for cpu in cpus},
        "mean_busy_pct": sum(values) / len(values),
        "peak_busy_pct": max(values),
        "load_one_recorded": load_one,
    }


def validate_performance_idle(evidence: dict[str, object]) -> None:
    mean_busy = evidence.get("mean_busy_pct")
    peak_busy = evidence.get("peak_busy_pct")
    if (not isinstance(mean_busy, (int, float)) or not math.isfinite(float(mean_busy)) or
            not isinstance(peak_busy, (int, float)) or not math.isfinite(float(peak_busy))):
        raise RuntimeError("CPU resource evidence is incomplete")
    if (float(mean_busy) >= PERFORMANCE_MEAN_BUSY_LIMIT_PCT or
            float(peak_busy) >= PERFORMANCE_PEAK_BUSY_LIMIT_PCT):
        raise RuntimeError(
            "formal performance requires the selected CPU set to be idle; "
            f"mean={float(mean_busy):.1f}% peak={float(peak_busy):.1f}%"
        )
