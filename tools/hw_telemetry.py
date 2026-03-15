#!/usr/bin/env python3
import glob
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RAPLDomain:
    name: str
    energy_path: str
    max_j: float
    source: str = "sysfs"
    energy_scale: float = 1.0


class TelemetryProbe:
    def __init__(self):
        self.cpu_domain: Optional[RAPLDomain] = None
        self.gpu_domain: Optional[RAPLDomain] = None
        self.gpu_rapl_nonfunctional = False
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self.nvidia_energy_supported = False
        self.windows_cpu_max_mhz = self._detect_windows_cpu_max_mhz() if os.name == "nt" else 0.0

        self._detect_rapl_domains()
        if self.has_nvidia_smi:
            self.nvidia_energy_supported = self._nvidia_query("energy") is not None

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    @staticmethod
    def _read_float(path: str, scale: float = 1.0) -> Optional[float]:
        txt = TelemetryProbe._read_text(path)
        if not txt:
            return None
        try:
            return float(txt) * scale
        except Exception:
            return None

    @staticmethod
    def _run_powershell(command: str) -> str:
        ps = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh.exe") or shutil.which("pwsh")
        if not ps:
            return ""
        try:
            res = subprocess.run(
                [ps, "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                return ""
            return (res.stdout or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _read_windows_counter(path: str, scale: float = 1.0) -> Optional[float]:
        safe_path = path.replace("'", "''")
        cmd = (
            "$ErrorActionPreference='Stop'; "
            f"$s=(Get-Counter -Counter '{safe_path}').CounterSamples | Select-Object -First 1; "
            "if ($s) { [Console]::Out.Write([System.Convert]::ToString([double]$s.CookedValue, [System.Globalization.CultureInfo]::InvariantCulture)) }"
        )
        txt = TelemetryProbe._run_powershell(cmd)
        if not txt:
            return None
        try:
            return float(txt) * scale
        except Exception:
            return None

    def _detect_windows_energy_domains(self) -> None:
        candidates = [
            ("RAPL_Package0_PKG", r"\Energy Meter(RAPL_Package0_PKG)\Energy"),
            ("_Total", r"\Energy Meter(_Total)\Energy"),
        ]
        for name, energy_path in candidates:
            if self._read_windows_counter(energy_path, scale=1e-9) is not None:
                self.cpu_domain = RAPLDomain(
                    name=name,
                    energy_path=energy_path,
                    max_j=0.0,
                    source="windows_counter",
                    energy_scale=1e-9,
                )
                break

    @staticmethod
    def _windows_average_cpu_property_mhz(prop_name: str) -> Optional[float]:
        cmd = (
            "$ErrorActionPreference='Stop'; "
            f"$m=Get-CimInstance Win32_Processor | Measure-Object -Property {prop_name} -Average; "
            "if ($m -and $m.Average) { "
            "[Console]::Out.Write([System.Convert]::ToString([double]$m.Average, [System.Globalization.CultureInfo]::InvariantCulture)) }"
        )
        txt = TelemetryProbe._run_powershell(cmd)
        if not txt:
            return None
        try:
            return float(txt)
        except Exception:
            return None

    @classmethod
    def _detect_windows_cpu_max_mhz(cls) -> float:
        v = cls._windows_average_cpu_property_mhz("MaxClockSpeed")
        return float(v or 0.0)

    def _detect_rapl_domains(self) -> None:
        if os.name == "nt":
            self._detect_windows_energy_domains()
            return
        paths = sorted(set(glob.glob("/sys/class/powercap/intel-rapl:*") + glob.glob("/sys/class/powercap/intel-rapl:*:*")))
        domains = []
        for p in paths:
            if not os.path.isdir(p):
                continue
            ep = os.path.join(p, "energy_uj")
            if not os.path.exists(ep):
                continue
            name = self._read_text(os.path.join(p, "name"))
            max_j = self._read_float(os.path.join(p, "max_energy_range_uj"), scale=1e-6) or 0.0
            domains.append(RAPLDomain(name=name, energy_path=ep, max_j=max_j))

        for d in domains:
            if d.name.startswith("package"):
                self.cpu_domain = d
                break
        if self.cpu_domain is None:
            for d in domains:
                if d.name == "core":
                    self.cpu_domain = d
                    break

        gpu_keywords = ("gpu", "uncore", "gt", "graphics")
        for d in domains:
            n = d.name.lower()
            if any(k in n for k in gpu_keywords):
                self.gpu_domain = d
                break

        if self.gpu_domain is not None and not self._verify_gpu_rapl_functional():
            self.gpu_domain = None
            self.gpu_rapl_nonfunctional = True

    def _verify_gpu_rapl_functional(self) -> bool:
        """Check if gpu_domain RAPL gives meaningful readings (>1mW change in 0.5s)."""
        if self.gpu_domain is None:
            return False
        e1 = self._read_rapl_j(self.gpu_domain)
        if e1 is None:
            return False
        time.sleep(0.3)
        e2 = self._read_rapl_j(self.gpu_domain)
        if e2 is None:
            return False
        delta = abs(e2 - e1)
        return delta > 0.001

    def _nvidia_query(self, key: str) -> Optional[float]:
        if not self.has_nvidia_smi:
            return None
        try:
            res = subprocess.run(
                ["nvidia-smi", f"--query-gpu={key}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                return None
            line = (res.stdout or "").strip().splitlines()
            if not line:
                return None
            return float(line[0].split(",")[0].strip())
        except Exception:
            return None

    def _read_rapl_j(self, domain: Optional[RAPLDomain]) -> Optional[float]:
        if domain is None:
            return None
        if domain.source == "windows_counter":
            return self._read_windows_counter(domain.energy_path, scale=domain.energy_scale)
        v_uj = self._read_float(domain.energy_path, scale=1.0)
        if v_uj is None:
            return None
        return v_uj * 1e-6

    @staticmethod
    def _linux_cpu_avg_freq_mhz() -> float:
        vals = []
        patterns = [
            "/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq",
            "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq",
            "/sys/devices/system/cpu/cpufreq/policy*/cpuinfo_cur_freq",
            "/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_cur_freq",
        ]
        for pattern in patterns:
            for p in glob.glob(pattern):
                v = TelemetryProbe._read_float(p, scale=1e-3)
                if v is not None and v > 0.0:
                    vals.append(v)
            if vals:
                break
        if not vals:
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    matches = re.findall(r"^cpu MHz\s*:\s*([0-9]+(?:\.[0-9]+)?)$", f.read(), flags=re.MULTILINE)
                vals = [float(v) for v in matches if float(v) > 0.0]
            except Exception:
                vals = []
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def _windows_cpu_avg_freq_mhz(self) -> float:
        counter = self._read_windows_counter(r"\Processor Information(_Total)\Processor Frequency")
        if counter is not None and counter > 0.0:
            return float(counter)

        current = self._windows_average_cpu_property_mhz("CurrentClockSpeed")
        if current is not None and current > 0.0:
            return float(current)

        perf_pct = self._read_windows_counter(r"\Processor Information(_Total)\% Processor Performance")
        if perf_pct is not None and perf_pct > 0.0 and self.windows_cpu_max_mhz > 0.0:
            return float(self.windows_cpu_max_mhz * perf_pct / 100.0)

        return 0.0

    def _cpu_avg_freq_mhz(self) -> float:
        if os.name == "nt":
            return self._windows_cpu_avg_freq_mhz()
        return self._linux_cpu_avg_freq_mhz()

    def _gpu_freq_mhz(self) -> float:
        v = self._read_float("/sys/class/drm/card0/gt_cur_freq_mhz", scale=1.0)
        if v is not None:
            return v
        nv = self._nvidia_query("clocks.current.graphics")
        return nv or 0.0

    def describe_sources(self) -> str:
        if self.cpu_domain:
            cpu_src = ("win_energy_meter:" if self.cpu_domain.source == "windows_counter" else "rapl:") + self.cpu_domain.name
        else:
            cpu_src = "none"
        if self.gpu_domain:
            gpu_src = ("win_energy_meter:" if self.gpu_domain.source == "windows_counter" else "rapl:") + self.gpu_domain.name
        elif self.nvidia_energy_supported:
            gpu_src = "nvidia:energy"
        elif self.has_nvidia_smi:
            gpu_src = "nvidia:power_draw"
        elif self.gpu_rapl_nonfunctional:
            gpu_src = "pkg_minus_idle"
        else:
            gpu_src = "none"
        return f"cpu={cpu_src};gpu={gpu_src}"

    @staticmethod
    def _delta_with_wrap(start_v: Optional[float], end_v: Optional[float], max_v: float) -> float:
        if start_v is None or end_v is None:
            return 0.0
        if end_v >= start_v:
            return end_v - start_v
        if max_v > 0:
            return (max_v - start_v) + end_v
        return 0.0

    def measure_idle_pkg_power_w(self, duration_s: float = 1.0) -> float:
        """Measure idle package power over duration_s seconds. Returns watts."""
        if self.cpu_domain is None:
            return 0.0
        e1 = self._read_rapl_j(self.cpu_domain)
        if e1 is None:
            return 0.0
        time.sleep(duration_s)
        e2 = self._read_rapl_j(self.cpu_domain)
        if e2 is None:
            return 0.0
        delta = self._delta_with_wrap(e1, e2, self.cpu_domain.max_j)
        return delta / duration_s

    def snapshot(self) -> Dict[str, Optional[float]]:
        snap = {
            "ts": time.time(),
            "cpu_freq_mhz": self._cpu_avg_freq_mhz(),
            "gpu_freq_mhz": self._gpu_freq_mhz(),
            "cpu_energy_j": self._read_rapl_j(self.cpu_domain),
            "gpu_energy_j": None,
            "gpu_power_w": None,
        }

        if self.gpu_domain is not None:
            snap["gpu_energy_j"] = self._read_rapl_j(self.gpu_domain)
        elif self.nvidia_energy_supported:
            mj = self._nvidia_query("energy")
            if mj is not None:
                snap["gpu_energy_j"] = mj * 1e-3
        elif self.has_nvidia_smi:
            snap["gpu_power_w"] = self._nvidia_query("power.draw")

        return snap

    def diff(self, start: Dict[str, Optional[float]], end: Dict[str, Optional[float]]) -> Dict[str, float]:
        elapsed = max(1e-9, float((end.get("ts") or 0.0) - (start.get("ts") or 0.0)))

        cpu_energy = self._delta_with_wrap(
            start.get("cpu_energy_j"),
            end.get("cpu_energy_j"),
            self.cpu_domain.max_j if self.cpu_domain else 0.0,
        )

        gpu_energy = 0.0
        if self.gpu_domain is not None:
            gpu_energy = self._delta_with_wrap(
                start.get("gpu_energy_j"),
                end.get("gpu_energy_j"),
                self.gpu_domain.max_j,
            )
        elif self.nvidia_energy_supported:
            gpu_energy = self._delta_with_wrap(start.get("gpu_energy_j"), end.get("gpu_energy_j"), 0.0)
        elif self.has_nvidia_smi:
            p0 = start.get("gpu_power_w")
            p1 = end.get("gpu_power_w")
            if p0 is not None and p1 is not None:
                gpu_energy = ((p0 + p1) * 0.5) * elapsed

        return {
            "elapsed_s": float(elapsed),
            "cpu_freq_start_mhz": float(start.get("cpu_freq_mhz") or 0.0),
            "cpu_freq_end_mhz": float(end.get("cpu_freq_mhz") or 0.0),
            "gpu_freq_start_mhz": float(start.get("gpu_freq_mhz") or 0.0),
            "gpu_freq_end_mhz": float(end.get("gpu_freq_mhz") or 0.0),
            "cpu_energy_j": float(cpu_energy),
            "gpu_energy_j": float(gpu_energy),
        }


def apply_freq_percent(control_script_path: str, percent: Optional[int]) -> str:
    if percent is None:
        return "not_requested"
    if percent < 0 or percent > 100:
        return f"invalid:{percent}"
    if not os.path.exists(control_script_path):
        return "missing_script"
    if os.name == "nt":
        return "unsupported_on_windows"

    cmd = [control_script_path, "freq", str(percent)]
    if os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return "ok"
        msg = ((res.stderr or "") + "\n" + (res.stdout or "")).strip().splitlines()
        short = msg[0][:80] if msg else ""
        return f"failed:{res.returncode}:{short}"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def apply_freq_mhz(control_script_path: str, mhz: Optional[int]) -> str:
    """Set frequency using absolute MHz value via control script's freq_mhz command."""
    if mhz is None:
        return "not_requested"
    if mhz < 1:
        return f"invalid:{mhz}"
    if not os.path.exists(control_script_path):
        return "missing_script"
    if os.name == "nt":
        return "unsupported_on_windows"

    cmd = [control_script_path, "freq_mhz", str(mhz)]
    if os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return "ok"
        msg = ((res.stderr or "") + "\n" + (res.stdout or "")).strip().splitlines()
        short = msg[0][:80] if msg else ""
        return f"failed:{res.returncode}:{short}"
    except Exception as exc:
        return f"error:{type(exc).__name__}"
