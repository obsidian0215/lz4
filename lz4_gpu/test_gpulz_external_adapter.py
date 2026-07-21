#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpulz_external_adapter as adapter


VALID_OUTPUT = """compression ratio: 2.5
compression e2e throughput: 3.25 GB/s
decompression e2e throughput: 8.5 GB/s
"""


class GPULZExternalAdapterTest(unittest.TestCase):
    def test_parse_output_and_convert_ratio(self) -> None:
        parsed = adapter.parse_output(VALID_OUTPUT, "")
        self.assertEqual(parsed["original_over_compressed_ratio"], 2.5)
        self.assertEqual(100.0 / parsed["original_over_compressed_ratio"], 40.0)

    def test_duplicate_or_failure_output_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            adapter.parse_output(VALID_OUTPUT + "compression ratio: 2.5\n", "")
        with self.assertRaises(RuntimeError):
            adapter.parse_output(VALID_OUTPUT, "verification failed!!! Index 3 is wrong")
        with self.assertRaises(RuntimeError):
            adapter.parse_output(VALID_OUTPUT.replace("3.25", "0"), "")

    def test_registered_sample_and_uint32_support(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = root / "sample.bin"
            sample.write_bytes(b"abcdefgh")
            digest = hashlib.sha256(sample.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema": "heterolz.samples.v1",
                "selection_name": "test",
                "files": [{"relative_path": sample.name, "size": 8, "sha256": digest}],
            }), encoding="utf-8")
            identity = adapter.load_registered_sample(sample, root, manifest)
            self.assertEqual(adapter.support_status(identity), (True, None))

            sample.write_bytes(b"abcdefg")
            manifest.write_text(json.dumps({
                "schema": "heterolz.samples.v1",
                "files": [{
                    "relative_path": sample.name,
                    "size": 7,
                    "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                }],
            }), encoding="utf-8")
            identity = adapter.load_registered_sample(sample, root, manifest)
            self.assertEqual(
                adapter.support_status(identity),
                (False, "input_size_not_multiple_of_uint32"),
            )

    def test_coverage_records_supported_and_unsupported_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(json.dumps({
                "schema": "heterolz.samples.v1",
                "selection_name": "paper",
                "files": [
                    {"relative_path": "ok", "size": 8, "sha256": "a" * 64},
                    {"relative_path": "tail", "size": 7, "sha256": "b" * 64},
                ],
            }), encoding="utf-8")
            coverage = adapter.build_coverage(manifest)
        self.assertEqual(coverage["total"], 2)
        self.assertEqual(coverage["supported"], 1)
        self.assertEqual(coverage["unsupported"], 1)
        self.assertEqual(
            coverage["files"][1]["reason"],
            "input_size_not_multiple_of_uint32",
        )

    def test_source_identity_binds_commit_remote_tree_and_clean_state(self) -> None:
        responses = {
            ("rev-parse", "HEAD"): adapter.UPSTREAM_COMMIT,
            ("remote", "get-url", "origin"): adapter.UPSTREAM_REPOSITORY,
            ("status", "--porcelain", "--untracked-files=no"): "",
            ("rev-parse", "HEAD^{tree}"): "a" * 40,
        }
        with tempfile.TemporaryDirectory() as temp, unittest.mock.patch.object(
            adapter, "run_git", side_effect=lambda root, *args: responses[args]
        ):
            result = adapter.inspect_source(Path(temp))
        self.assertEqual(result["commit"], adapter.UPSTREAM_COMMIT)
        self.assertEqual(result["git_tree"], "a" * 40)

        responses[("status", "--porcelain", "--untracked-files=no")] = " M gpulz.cu"
        with tempfile.TemporaryDirectory() as temp, unittest.mock.patch.object(
            adapter, "run_git", side_effect=lambda root, *args: responses[args]
        ), self.assertRaises(ValueError):
            adapter.inspect_source(Path(temp))

    def test_cuda_identity_binds_toolkit_driver_and_gpu(self) -> None:
        nvcc_output = "Cuda compilation tools, release 11.4, V11.4.120\n"
        smi_output = "0, Quadro P400, GPU-abcd, 00000000:02:00.0, 545.29.02\n"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nvcc = root / "nvcc"
            smi = root / "nvidia-smi"
            nvcc.write_bytes(b"nvcc")
            smi.write_bytes(b"smi")
            with unittest.mock.patch.object(
                adapter,
                "run_identity_tool",
                side_effect=[nvcc_output, smi_output],
            ):
                identity = adapter.inspect_cuda_environment(nvcc, smi, 0, "sm_61")
        self.assertEqual(identity["cuda_release"], "11.4")
        self.assertEqual(identity["cuda_arch"], "sm_61")
        self.assertEqual(identity["gpu"]["name"], "Quadro P400")
        self.assertEqual(identity["gpu"]["driver_version"], "545.29.02")

    def test_cuda_identity_rejects_bad_arch_or_ambiguous_gpu(self) -> None:
        with self.assertRaises(ValueError):
            adapter.inspect_cuda_environment(Path("missing"), Path("missing"), 0, "compute_61")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nvcc = root / "nvcc"
            smi = root / "nvidia-smi"
            nvcc.write_bytes(b"nvcc")
            smi.write_bytes(b"smi")
            with unittest.mock.patch.object(
                adapter,
                "run_identity_tool",
                side_effect=[
                    "Cuda compilation tools, release 11.4, V11.4.120\n",
                    "1, Tesla K20c, GPU-efgh, 00000000:04:00.0, 470.256.02\n",
                ],
            ), self.assertRaises(RuntimeError):
                adapter.inspect_cuda_environment(nvcc, smi, 0, "sm_35")

    def test_execute_uses_fixed_upstream_cli_and_records_cold_time(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["gpulz"], returncode=0, stdout=VALID_OUTPUT, stderr=""
        )
        with unittest.mock.patch.object(adapter.subprocess, "run", return_value=completed) as run, \
                unittest.mock.patch.object(adapter.time, "perf_counter_ns", side_effect=[100, 250]):
            result, elapsed = adapter.execute(
                Path("gpulz"), Path("registered.bin"), Path("."), 5.0, {"A": "B"}
            )
        self.assertIs(result, completed)
        self.assertEqual(elapsed, 150)
        self.assertEqual(run.call_args.args[0], ["gpulz", "-i", "registered.bin"])

    def test_execute_with_fake_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / "fake_gpulz.py"
            sample = root / "registered.bin"
            sample.write_bytes(b"abcd")
            fake.write_text(
                "import pathlib, sys\n"
                "assert sys.argv[1] == '-i'\n"
                "assert pathlib.Path(sys.argv[2]).read_bytes() == b'abcd'\n"
                "print('compression ratio: 2.5')\n"
                "print('compression e2e throughput: 3.25 GB/s')\n"
                "print('decompression e2e throughput: 8.5 GB/s')\n",
                encoding="utf-8",
            )
            completed, elapsed = adapter.execute(
                Path(sys.executable), sample, root, 5.0, binary_args=(str(fake),)
            )
            self.assertEqual(completed.returncode, 0)
            self.assertGreater(elapsed, 0)
            self.assertEqual(
                adapter.parse_output(completed.stdout, completed.stderr)[
                    "compression_pipeline_gbps"
                ],
                3.25,
            )

    def test_output_paths_reject_aliases_and_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = root / "a"
            b = root / "b"
            c = root / "c"
            adapter.ensure_output_paths((a, b, c), ())
            a.write_text("occupied", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                adapter.ensure_output_paths((a, b, c), ())
            with self.assertRaises(ValueError):
                adapter.ensure_output_paths((b, b, c), ())


if __name__ == "__main__":
    unittest.main()
