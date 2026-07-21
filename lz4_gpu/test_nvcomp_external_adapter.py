#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvcomp_external_adapter as adapter


def valid_output(input_size: int = 131073) -> str:
    batch_size = (input_size - 1) // adapter.CHUNK_SIZE + 1
    return "\n".join([
        "schema=heterolz.nvcomp-lz4-batched.v1",
        "api=low-level-batched-cpp",
        "algorithm=LZ4",
        "data_type=char",
        "bitshuffle_mode=0",
        "decompress_backend=0",
        f"input_bytes={input_size}",
        "compressed_bytes=40000",
        f"chunk_size={adapter.CHUNK_SIZE}",
        f"batch_size={batch_size}",
        f"warmup={adapter.WARMUP}",
        f"iterations={adapter.ITERATIONS}",
        f"stream={adapter.STREAM_MODE}",
        f"synchronization={adapter.SYNCHRONIZATION_MODE}",
        f"warm_e2e_scope={adapter.WARM_E2E_SCOPE}",
        "compression_kernel_ms=0.25",
        "decompression_kernel_ms=0.125",
        "compression_warm_e2e_ns=500000",
        "decompression_warm_e2e_ns=250000",
        "roundtrip_ok=true",
        "",
    ])


class NvcompExternalAdapterTest(unittest.TestCase):
    def test_parse_output_binds_cpp_batched_configuration(self) -> None:
        result = adapter.parse_output(valid_output(), "", 131073)
        self.assertEqual(result["batch_size"], 3)
        self.assertEqual(result["compressed_bytes"], 40000)
        self.assertEqual(result["compression_kernel_ms"], 0.25)

    def test_parse_output_rejects_duplicate_failure_and_wrong_chunking(self) -> None:
        output = valid_output()
        with self.assertRaises(RuntimeError):
            adapter.parse_output(output + "input_bytes=131073\n", "", 131073)
        with self.assertRaises(RuntimeError):
            adapter.parse_output(output, "CUDA error: illegal memory access", 131073)
        with self.assertRaises(RuntimeError):
            adapter.parse_output(output.replace("batch_size=3", "batch_size=2"), "", 131073)
        with self.assertRaises(RuntimeError):
            adapter.parse_output(output.replace("decompress_backend=0", "decompress_backend=2"), "", 131073)

    def test_package_identity_requires_fixed_version_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = root / "lz4.h"
            library = root / "libnvcomp.so.5"
            version = root / "nvcomp-config-version.cmake"
            license_file = root / "LICENSE"
            header.write_text("header\n", encoding="utf-8")
            library.write_bytes(b"library")
            version.write_text('set(PACKAGE_VERSION "5.3.0")\n', encoding="utf-8")
            license_file.write_text("license\n", encoding="utf-8")
            identity = adapter.inspect_nvcomp_package(
                header, library, version, license_file
            )
            self.assertEqual(identity["version"], "5.3.0")
            self.assertEqual(len(identity["library_sha256"]), 64)
            version.write_text('set(PACKAGE_VERSION "5.2.0")\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                adapter.inspect_nvcomp_package(header, library, version, license_file)

    def test_dynamic_linkage_must_resolve_registered_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ldd = root / "ldd"
            binary = root / "nvcomp_lz4_batched"
            library = root / "libnvcomp.so.5"
            for path in (ldd, binary, library):
                path.write_bytes(path.name.encode())
            completed = subprocess.CompletedProcess(
                args=[str(ldd), str(binary)],
                returncode=0,
                stdout=f"libnvcomp.so.5 => {library} (0x1234)\n",
                stderr="",
            )
            with unittest.mock.patch.object(
                adapter.subprocess, "run", return_value=completed
            ):
                identity = adapter.inspect_dynamic_linkage(ldd, binary, library)
            self.assertEqual(
                identity["resolved_nvcomp_library"], str(library.resolve())
            )

            other = root / "other" / "libnvcomp.so.5"
            other.parent.mkdir()
            other.write_bytes(b"other")
            completed.stdout = f"libnvcomp.so.5 => {other} (0x1234)\n"
            with unittest.mock.patch.object(
                adapter.subprocess, "run", return_value=completed
            ), self.assertRaises(ValueError):
                adapter.inspect_dynamic_linkage(ldd, binary, library)

    def test_execute_uses_fixed_formal_arguments(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["binary"], returncode=0, stdout=valid_output(4), stderr=""
        )
        with unittest.mock.patch.object(
            adapter.subprocess, "run", return_value=completed
        ) as run, unittest.mock.patch.object(
            adapter.time, "perf_counter_ns", side_effect=[100, 250]
        ):
            result, elapsed = adapter.execute(
                Path("binary"), Path("registered.bin"), 2, 5.0
            )
        self.assertIs(result, completed)
        self.assertEqual(elapsed, 150)
        self.assertEqual(run.call_args.args[0], [
            "binary", "--input", "registered.bin",
            "--chunk-size", "65536", "--warmup", "3",
            "--iterations", "1", "--gpu-index", "2",
            "--decompress-backend", "0",
        ])

    def test_throughput_uses_decimal_gigabytes(self) -> None:
        self.assertEqual(adapter.throughput_gbps(1_000_000_000, 1_000_000_000), 1.0)


if __name__ == "__main__":
    unittest.main()
