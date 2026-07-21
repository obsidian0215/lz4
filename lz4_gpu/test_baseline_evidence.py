#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_evidence as evidence
import gpulz_external_adapter as adapter
import nvcomp_external_adapter as nvcomp


class BaselineEvidenceTest(unittest.TestCase):
    def make_inputs(self, root: Path, repetitions: int = 2) -> tuple[Path, Path, list[Path]]:
        manifest = root / "paper.json"
        manifest.write_text(json.dumps({
            "schema": "heterolz.samples.v1",
            "selection_name": "paper-test",
            "files": [
                {"relative_path": "supported", "size": 8, "sha256": "a" * 64},
                {"relative_path": "unsupported", "size": 7, "sha256": "b" * 64},
            ],
        }) + "\n", encoding="utf-8")
        coverage = root / "gpulz-coverage.json"
        coverage.write_text(
            json.dumps(adapter.build_coverage(manifest)) + "\n", encoding="utf-8"
        )
        manifest_sha = evidence.sha256_file(manifest)
        stdout = (
            "compression ratio: 2.5\n"
            "compression e2e throughput: 3.25 GB/s\n"
            "decompression e2e throughput: 8.5 GB/s\n"
        )
        records: list[Path] = []
        for repetition in range(1, repetitions + 1):
            record = {
                "schema": adapter.SCHEMA,
                "status": "complete",
                "repetition": repetition,
                "baseline": {
                    "name": "GPULZ",
                    "repository": adapter.UPSTREAM_REPOSITORY,
                    "origin": adapter.UPSTREAM_REPOSITORY,
                    "commit": adapter.UPSTREAM_COMMIT,
                    "git_tree": "c" * 40,
                    "distribution_mode": "external-fixed-artifact",
                    "license_status": "no-license-file-at-fixed-commit",
                    "binary_sha256": "d" * 64,
                    "parameters": adapter.UPSTREAM_PARAMETERS,
                },
                "environment": {
                    "cuda_arch": "sm_61",
                    "cuda_release": "11.4",
                    "nvcc_path": "/usr/local/cuda/bin/nvcc",
                    "nvcc_sha256": "e" * 64,
                    "nvidia_smi_path": "/usr/bin/nvidia-smi",
                    "nvidia_smi_sha256": "f" * 64,
                    "gpu": {
                        "index": 0,
                        "name": "Quadro P400",
                        "uuid": "GPU-test",
                        "pci_bus_id": "00000000:02:00.0",
                        "driver_version": "545.29.02",
                    },
                    "cuda_visible_devices": None,
                },
                "sample": {
                    "relative_path": "supported",
                    "size": 8,
                    "sha256": "a" * 64,
                    "manifest_sha256": manifest_sha,
                    "selection_name": "paper-test",
                },
                "support": {"supported": True, "reason": None},
                "correctness_scope": "upstream_internal_roundtrip_only",
                "compressed_size_exact_available": False,
                "process": {
                    "returncode": 0,
                    "cold_total_ns": 1000 + repetition,
                    "stdout": stdout,
                    "stderr": "",
                    "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                },
                "timing": {
                    "upstream_compression_pipeline_gbps": 3.25,
                    "upstream_decompression_pipeline_gbps": 8.5,
                    "upstream_boundary": (
                        "CUDA events after input H2D and before result D2H; "
                        "compression and decompression execute in one process"
                    ),
                    "process_cold_total_ns": 1000 + repetition,
                },
                "compression": {
                    "upstream_original_over_compressed_ratio": 2.5,
                    "compressed_over_original_percent": 40.0,
                },
                "verification": {
                    "upstream_failure_text_absent": True,
                    "input_tail_fully_covered": True,
                },
            }
            path = root / f"record-{repetition}.json"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            records.append(path)
        return manifest, coverage, records

    def make_nvcomp_records(self, root: Path, manifest: Path,
                            repetitions: int = 2) -> list[Path]:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_sha = evidence.sha256_file(manifest)
        baseline = {
            "name": "nvCOMP",
            "version": nvcomp.NVCOMP_VERSION,
            "distribution_mode": "external-fixed-binary-package",
            "official_samples_repository": nvcomp.UPSTREAM_SAMPLES_REPOSITORY,
            "official_samples_commit": nvcomp.UPSTREAM_SAMPLES_COMMIT,
            "header_path": "/opt/nvcomp/include/nvcomp/lz4.h",
            "header_sha256": "1" * 64,
            "library_path": "/opt/nvcomp/lib/libnvcomp.so.5",
            "library_sha256": "2" * 64,
            "version_file_path": "/opt/nvcomp/lib/cmake/nvcomp-config-version.cmake",
            "version_file_sha256": "3" * 64,
            "license_file_path": "/opt/nvcomp/LICENSE",
            "license_file_sha256": "4" * 64,
            "ldd_path": "/usr/bin/ldd",
            "ldd_sha256": "5" * 64,
            "resolved_nvcomp_library": "/opt/nvcomp/lib/libnvcomp.so.5",
            "ldd_stdout_sha256": "6" * 64,
            "binary_path": "/tmp/nvcomp_lz4_batched",
            "binary_sha256": "7" * 64,
            "driver_source_path": "/root/heterolz-formal/lz4_gpu/nvcomp_lz4_batched.cu",
            "driver_source_sha256": "8" * 64,
            "api": "low-level-batched-cpp",
            "algorithm": "LZ4",
            "parameters": {
                "chunk_size": nvcomp.CHUNK_SIZE,
                "data_type": "char",
                "bitshuffle_mode": 0,
                "decompress_backend": nvcomp.DECOMPRESS_BACKEND,
                "warmup": nvcomp.WARMUP,
                "iterations": nvcomp.ITERATIONS,
                "stream": nvcomp.STREAM_MODE,
                "synchronization": nvcomp.SYNCHRONIZATION_MODE,
                "warm_e2e_scope": nvcomp.WARM_E2E_SCOPE,
            },
        }
        environment = {"cuda_arch": "sm_89", "gpu": {"uuid": "GPU-test"}}
        records: list[Path] = []
        for sample in manifest_data["files"]:
            for repetition in range(1, repetitions + 1):
                stdout = f"sample={sample['relative_path']} repetition={repetition}\n"
                input_bytes = sample["size"]
                comp_ms = 1.0
                decomp_ms = 0.5
                comp_warm_ns = 2_000_000
                decomp_warm_ns = 1_000_000
                record = {
                    "schema": nvcomp.SCHEMA,
                    "status": "complete",
                    "repetition": repetition,
                    "baseline": baseline,
                    "environment": environment,
                    "sample": {**sample, "manifest_sha256": manifest_sha,
                               "selection_name": manifest_data["selection_name"]},
                    "process": {
                        "returncode": 0,
                        "cold_total_ns": 3_000_000,
                        "stdout": stdout,
                        "stderr": "",
                        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                    },
                    "chunking": {"chunk_size": nvcomp.CHUNK_SIZE, "batch_size": 1},
                    "compression": {
                        "compressed_bytes": 4,
                        "compressed_over_original_percent": 4 / input_bytes * 100.0,
                    },
                    "timing": {
                        "compression_kernel_ms": comp_ms,
                        "decompression_kernel_ms": decomp_ms,
                        "compression_warm_e2e_ns": comp_warm_ns,
                        "decompression_warm_e2e_ns": decomp_warm_ns,
                        "process_cold_total_ns": 3_000_000,
                        "compression_kernel_gbps": input_bytes / (comp_ms * 1_000_000),
                        "decompression_kernel_gbps": input_bytes / (decomp_ms * 1_000_000),
                        "compression_warm_e2e_gbps": input_bytes / comp_warm_ns,
                        "decompression_warm_e2e_gbps": input_bytes / decomp_warm_ns,
                    },
                    "verification": {
                        "roundtrip": "passed",
                        "scope": "nvcomp_internal_batched_roundtrip",
                    },
                }
                path = root / f"nvcomp-{sample['relative_path']}-{repetition}.json"
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                records.append(path)
        return records

    def test_build_contract_preserves_records_and_unsupported_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, source_coverage, records = self.make_inputs(Path(temp))
            registry, coverage = evidence.build_gpulz_contract(
                manifest, source_coverage, records, 2, formal=False
            )
        self.assertEqual(registry["schema"], evidence.REGISTRY_SCHEMA)
        self.assertEqual(len(registry["baselines"][0]["records"]), 2)
        files = {item["relative_path"]: item for item in coverage["baselines"][0]["files"]}
        self.assertEqual(files["supported"]["completed_repetitions"], [1, 2])
        self.assertEqual(files["unsupported"]["status"], "unsupported")
        self.assertEqual(files["unsupported"]["completed_repetitions"], [])

    def test_missing_or_duplicate_repetitions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, source_coverage, records = self.make_inputs(Path(temp))
            with self.assertRaises(ValueError):
                evidence.build_gpulz_contract(
                    manifest, source_coverage, records[:1], 2, formal=False
                )
            with self.assertRaises(ValueError):
                evidence.build_gpulz_contract(
                    manifest, source_coverage, [records[0], records[0]], 2, formal=False
                )

    def test_raw_output_and_sample_identity_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, source_coverage, records = self.make_inputs(root)
            record = json.loads(records[0].read_text(encoding="utf-8"))
            record["process"]["stdout"] += "changed"
            records[0].write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                evidence.build_gpulz_contract(
                    manifest, source_coverage, records, 2, formal=False
                )

    def test_formal_mode_rejects_noncanonical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, source_coverage, records = self.make_inputs(Path(temp), repetitions=9)
            with self.assertRaises(ValueError):
                evidence.build_gpulz_contract(
                    manifest, source_coverage, records, 9, formal=True
                )

    def test_nvcomp_records_cover_every_paper_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, source_coverage, records = self.make_inputs(root)
            nvcomp_records = self.make_nvcomp_records(root, manifest)
            registry, coverage = evidence.build_gpulz_contract(
                manifest, source_coverage, records, 2, formal=False,
                nvcomp_record_paths=nvcomp_records,
            )
            self.assertEqual(
                [item["name"] for item in registry["baselines"]],
                ["GPULZ", "nvCOMP"],
            )
            self.assertEqual(coverage["baselines"][1]["measured"], 2)
            with self.assertRaises(ValueError):
                evidence.build_gpulz_contract(
                    manifest, source_coverage, records, 2, formal=False,
                    nvcomp_record_paths=nvcomp_records[:-1],
                )


if __name__ == "__main__":
    unittest.main()
