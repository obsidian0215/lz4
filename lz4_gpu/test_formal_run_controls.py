#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_formal_acceptance as formal
import benchmark_real_samples as benchmark
import formal_cpu_budget as budget


ADMISSION_RUN_ID = "heterolz-admission-20260719T120000000000Z"


class FormalRunControlsTest(unittest.TestCase):
    def test_registered_source_must_match_exactly(self) -> None:
        identity = {
            "schema": "heterolz.source-fingerprint.v1",
            "git_commit": "a" * 40,
            "source_fingerprint": "b" * 64,
            "files": [{"path": "lz4_gpu/lz4_gpu.c", "size": 1, "sha256": "c" * 64}],
        }
        with tempfile.TemporaryDirectory() as temp:
            registry = Path(temp) / "formal_source_fingerprint.json"
            registry.write_text(json.dumps(identity) + "\n", encoding="utf-8")
            formal.validate_registered_source(identity, registry)

            changed = dict(identity)
            changed["git_commit"] = "d" * 40
            with self.assertRaises(ValueError):
                formal.validate_registered_source(changed, registry)

            registry.write_text(json.dumps({"schema": "wrong"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                formal.validate_registered_source(identity, registry)

    def test_cpu_set_parser_and_performance_idle_gate(self) -> None:
        self.assertEqual(budget.parse_cpu_set("0,2-4"), (0, 2, 3, 4))
        for value in ("", "0,0", "2-1", "x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget.parse_cpu_set(value)
        budget.validate_performance_idle({"mean_busy_pct": 10.0, "peak_busy_pct": 20.0})
        for evidence in (
            {"mean_busy_pct": 15.0, "peak_busy_pct": 20.0},
            {"mean_busy_pct": 10.0, "peak_busy_pct": 35.0},
            {"mean_busy_pct": math.nan, "peak_busy_pct": 0.0},
        ):
            with self.assertRaises(RuntimeError):
                budget.validate_performance_idle(evidence)

    def test_cpu_topology_requires_distinct_physical_cores(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for cpu, core in ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3)):
                topology = root / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text("0\n", encoding="utf-8")
                (topology / "core_id").write_text(f"{core}\n", encoding="utf-8")
            self.assertEqual(budget.discover_cpu_set(4, root), (0, 2, 3, 4))
            self.assertEqual(
                budget.collect_cpu_topology((0, 2, 3, 4), root),
                [
                    {"cpu": 0, "package": 0, "core": 0, "core_class_source": "uniform",
                     "core_class_id": None, "core_class": "uniform", "scaling_cur_freq_khz": None,
                     "scaling_min_freq_khz": None, "scaling_max_freq_khz": None,
                     "scaling_governor": None},
                    {"cpu": 2, "package": 0, "core": 1, "core_class_source": "uniform",
                     "core_class_id": None, "core_class": "uniform", "scaling_cur_freq_khz": None,
                     "scaling_min_freq_khz": None, "scaling_max_freq_khz": None,
                     "scaling_governor": None},
                    {"cpu": 3, "package": 0, "core": 2, "core_class_source": "uniform",
                     "core_class_id": None, "core_class": "uniform", "scaling_cur_freq_khz": None,
                     "scaling_min_freq_khz": None, "scaling_max_freq_khz": None,
                     "scaling_governor": None},
                    {"cpu": 4, "package": 0, "core": 3, "core_class_source": "uniform",
                     "core_class_id": None, "core_class": "uniform", "scaling_cur_freq_khz": None,
                     "scaling_min_freq_khz": None, "scaling_max_freq_khz": None,
                     "scaling_governor": None},
                ],
            )
            with self.assertRaises(RuntimeError):
                budget.collect_cpu_topology((0, 1, 2, 3), root)

    def test_cpu_discovery_prefers_one_package_performance_class(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = (
                (0, 0, 0, 1), (1, 0, 1, 1), (2, 0, 2, 1), (3, 0, 3, 1),
                (4, 0, 4, 2), (5, 0, 5, 2), (6, 0, 6, 2), (7, 0, 7, 2),
                (8, 1, 0, 2), (9, 1, 1, 2), (10, 1, 2, 2), (11, 1, 3, 2),
            )
            for cpu, package, core, core_type in records:
                topology = root / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text(f"{package}\n", encoding="utf-8")
                (topology / "core_id").write_text(f"{core}\n", encoding="utf-8")
                (topology / "core_type").write_text(f"{core_type}\n", encoding="utf-8")
            self.assertEqual(budget.discover_cpu_set(4, root), (4, 5, 6, 7))
            self.assertEqual(budget.discover_cpu_set(4, root, allowed_cpus=(8, 9, 10, 11)),
                             (8, 9, 10, 11))
            with self.assertRaises(RuntimeError):
                budget.collect_cpu_topology((4, 5, 6, 8), root)

    def test_cpu_environment_records_formal_and_discovery_sets(self) -> None:
        with unittest.mock.patch.dict(budget.os.environ, {}, clear=True):
            budget.apply_cpu_environment((0, 2, 4, 6), tuple(range(8)))
            self.assertEqual(budget.os.environ["HETEROLZ_CPU_SET"], "0,2,4,6")
            self.assertEqual(budget.os.environ["HETEROLZ_CPU_DISCOVERY_SET"], "0,1,2,3,4,5,6,7")
            with self.assertRaises(ValueError):
                budget.apply_cpu_environment((0, 9), tuple(range(8)))

    def test_device_matrix_restores_formal_affinity(self) -> None:
        calls: list[tuple[str, tuple[int, ...]]] = []
        affinities: list[tuple[int, ...]] = []
        with unittest.mock.patch.dict(formal.os.environ, {}, clear=True), \
                unittest.mock.patch.object(
                    formal, "apply_process_affinity",
                    side_effect=lambda cpus: affinities.append(tuple(cpus)),
                ), \
                unittest.mock.patch.object(
                    formal, "query_device_info",
                    side_effect=lambda binary, cwd, venue: calls.append((venue, ())) or {"device_type": venue},
                ):
            matrix = formal.query_device_matrix(
                Path("binary"), Path("cwd"), (0, 2, 4, 6), tuple(range(8))
            )
            self.assertEqual(formal.os.environ["HETEROLZ_CPU_SET"], "0,2,4,6")
        self.assertEqual(set(matrix), {"GPU", "CPU"})
        self.assertEqual([venue for venue, _ in calls], ["GPU", "CPU"])
        self.assertEqual(affinities, [tuple(range(8)), (0, 2, 4, 6)])

    def test_formal_repo_endpoint_is_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                formal.validate_formal_repo(Path(temp))

    def test_admission_must_be_direct_child_of_canonical_result_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            root.mkdir()
            admission = root / ADMISSION_RUN_ID
            admission.mkdir()
            self.assertEqual(formal.validate_admission_directory(admission, root.resolve()),
                             admission.resolve())

            performance = root / "heterolz-performance-20260719T120000000000Z"
            performance.mkdir()
            with self.assertRaises(ValueError):
                formal.validate_admission_directory(performance, root.resolve())

            outside = Path(temp) / "outside" / ADMISSION_RUN_ID
            outside.mkdir(parents=True)
            with self.assertRaises(ValueError):
                formal.validate_admission_directory(outside, root.resolve())

    def test_baseline_inputs_bind_paper_manifest_and_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paper = root / "paper.json"
            paper.write_text(json.dumps({
                "schema": "heterolz.samples.v1",
                "selection_name": "paper16",
                "files": [
                    {"relative_path": f"sample-{index}", "size": 4,
                     "sha256": f"{index:064x}"}
                    for index in range(16)
                ],
            }) + "\n", encoding="utf-8")
            paper_sha = formal.sha256_file(paper)
            registry = root / "registry.json"
            coverage = root / "coverage.json"
            common = {"sample_manifest_sha256": paper_sha,
                      "selection_name": "paper16", "repetitions": 9,
                      "baselines": [{"name": "GPULZ"}, {"name": "nvCOMP"}]}
            registry.write_text(json.dumps({
                "schema": "heterolz.baseline-registry.v1", **common
            }) + "\n", encoding="utf-8")
            coverage.write_text(json.dumps({
                "schema": "heterolz.baseline-coverage.v1", **common
            }) + "\n", encoding="utf-8")
            with unittest.mock.patch.object(
                formal, "CANON_PAPER_MANIFEST_SHA256", paper_sha
            ):
                formal.validate_baseline_inputs(paper, registry, coverage, 9)
                with self.assertRaises(ValueError):
                    formal.validate_baseline_inputs(paper, registry, coverage, 10)

    def test_policy_schedule_rotates_reproducibly(self) -> None:
        specs = [
            ("fixed_n1", 1, False, "heterolz", "GPU", 4),
            ("fixed_n2", 2, False, "heterolz", "GPU", 4),
            ("fixed_n4", 4, False, "heterolz", "GPU", 4),
            ("fixed_n8", 8, False, "heterolz", "GPU", 4),
            ("adaptive", None, True, "heterolz", "GPU", 4),
            ("native_4w", 1, False, "native_lz4", "CPU", 4),
        ]
        orders = [benchmark.rotated_policy_specs(specs, 0, repetition)
                  for repetition in range(1, 7)]
        self.assertEqual([order[0][0] for order in orders],
                         ["fixed_n1", "fixed_n2", "fixed_n4", "fixed_n8", "adaptive",
                          "native_4w"])
        self.assertTrue(all(sorted(order) == sorted(specs) for order in orders))
        self.assertEqual(benchmark.rotated_policy_specs(specs, 1, 1), orders[1])

    def test_measured_runner_writes_resource_record_and_raw_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resources = root / "resources.jsonl"
            runner = benchmark.Runner(
                root / "commands.log", root / "raw.stdout", resources,
                "GPU", root / "profile",
            )
            identity = {
                "sample": "silesia/dickens",
                "policy": "fixed_n1",
                "engine": "heterolz",
                "venue": "GPU",
                "workers": 4,
                "operation": "compress",
                "repetition": 1,
            }
            measurement = runner.run(
                [sys.executable, "-c", "print('measured')"],
                native=True,
                resource_identity=identity,
            )
            self.assertIsNotNone(measurement)
            record = json.loads(resources.read_text(encoding="utf-8"))
            self.assertEqual(record["schema"], "heterolz.resource-record.v1")
            self.assertEqual({name: record[name] for name in identity}, identity)
            self.assertEqual(record["measurement"], measurement)
            metric = {
                "input_bytes": 1024,
                "output_bytes": 512,
                "n": 1,
                "chunk_blocks": 1,
                "kernel_us": 10,
                "no_ocl_us": 20,
                "total_us": 30,
                "ratio_pct": 50,
            }
            row = benchmark.metric_row(
                "silesia/dickens", "fixed_n1", "heterolz", "GPU", 4,
                "compress", 1, metric, True, True, measurement,
            )
            self.assertEqual(row["original_bytes"], 1024)
            self.assertEqual(row["process_elapsed_us"], measurement["process_elapsed_us"])
            self.assertEqual(row["resource_scope"], "process_wall_with_machine_energy")


if __name__ == "__main__":
    unittest.main()
