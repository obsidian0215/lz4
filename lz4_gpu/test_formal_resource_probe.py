#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import formal_resource_probe as probe


class FormalResourceProbeTest(unittest.TestCase):
    def test_cpu_package_energy_delta_and_wrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            domain = root / "intel-rapl-0"
            domain.mkdir()
            (domain / "name").write_text("package-0\n", encoding="utf-8")
            (domain / "energy_uj").write_text("900\n", encoding="utf-8")
            (domain / "max_energy_range_uj").write_text("1000\n", encoding="utf-8")
            energy = probe.EnergyProbe(root, Path(temp) / "missing-nvidia-smi")
            self.assertIsNotNone(energy.cpu_package)
            result = energy.summarize([
                {"time_ns": 1, "cpu_package_energy_uj": 900.0,
                 "gpu_energy_mj": None, "gpu_power_w": None},
                {"time_ns": 2, "cpu_package_energy_uj": 100.0,
                 "gpu_energy_mj": None, "gpu_power_w": None},
                {"time_ns": 3, "cpu_package_energy_uj": 900.0,
                 "gpu_energy_mj": None, "gpu_power_w": None},
                {"time_ns": 4, "cpu_package_energy_uj": 100.0,
                 "gpu_energy_mj": None, "gpu_power_w": None},
            ])
            self.assertAlmostEqual(result["cpu_package_energy_j"], 0.0012)
            self.assertEqual(result["gpu_energy_j"], None)
            self.assertEqual(result["scope"], "process_wall_with_machine_energy")
            self.assertEqual(
                result["counter_metadata"], {"cpu_package_max_energy_uj": 1000.0}
            )

    def test_run_measured_collects_process_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            energy = probe.EnergyProbe(Path(temp) / "powercap", Path(temp) / "nvidia-smi")
            completed, result = probe.run_measured(
                [sys.executable, "-c",
                 "import time; data=bytearray(8*1024*1024); print(len(data)); time.sleep(0.08)"],
                interval_s=0.01,
                energy_probe=energy,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(completed.stdout.strip(), str(8 * 1024 * 1024))
            self.assertEqual(result["schema"], probe.SCHEMA)
            self.assertEqual(result["status"], "complete")
            self.assertGreater(result["process_elapsed_us"], 0)
            self.assertGreaterEqual(result["cpu_user_us"], 0)
            self.assertGreaterEqual(result["cpu_system_us"], 0)
            self.assertEqual(
                result["process_cpu_time_source"], "rusage_children_minus_probe_helpers"
            )
            if os.name != "nt":
                self.assertGreater(result["peak_rss_bytes"], 0)
            self.assertEqual(
                result["sources"],
                {"cpu_package": "unavailable", "gpu": "unavailable"},
            )
            json.dumps(result)

    def test_nonzero_process_preserves_output_and_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            energy = probe.EnergyProbe(Path(temp) / "powercap", Path(temp) / "nvidia-smi")
            completed, result = probe.run_measured(
                [sys.executable, "-c",
                 "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(7)"],
                energy_probe=energy,
            )
            self.assertEqual(completed.returncode, 7)
            self.assertEqual(completed.stdout.strip(), "out")
            self.assertEqual(completed.stderr.strip(), "err")
            self.assertEqual(result["process_returncode"], 7)
            self.assertGreater(result["process_elapsed_us"], 0)

    def test_nvidia_values_are_aggregated_and_helper_cpu_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / "nvidia-smi"
            executable.write_text("placeholder\n", encoding="utf-8")
            usage = iter(((1.0, 2.0), (1.25, 2.5)))
            completed = unittest.mock.Mock(returncode=0, stdout="10\n20\n")
            with unittest.mock.patch.object(probe, "_child_usage", side_effect=lambda: next(usage)), \
                    unittest.mock.patch.object(probe.subprocess, "run", return_value=completed):
                energy = probe.EnergyProbe.__new__(probe.EnergyProbe)
                energy.nvidia_smi = executable
                energy._helper_user_s = 0.0
                energy._helper_system_s = 0.0
                self.assertEqual(energy._read_nvidia("power.draw"), 30.0)
                self.assertEqual(energy.helper_usage(), (0.25, 0.5))


if __name__ == "__main__":
    unittest.main()
