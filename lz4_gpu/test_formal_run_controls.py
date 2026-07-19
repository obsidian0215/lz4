#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_formal_acceptance as formal
import benchmark_real_samples as benchmark


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

    def test_formal_load_gate_is_strict_and_finite(self) -> None:
        self.assertEqual(formal.validate_formal_load(0.0), 0.0)
        self.assertEqual(formal.validate_formal_load(0.49), 0.49)
        for value in (0.5, 1.0, -0.1, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                formal.validate_formal_load(value)

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

    def test_policy_schedule_rotates_reproducibly(self) -> None:
        specs = [("fixed_n1", 1, False), ("fixed_n2", 2, False),
                 ("fixed_n4", 4, False), ("fixed_n8", 8, False),
                 ("adaptive", None, True)]
        orders = [benchmark.rotated_policy_specs(specs, 0, repetition)
                  for repetition in range(1, 6)]
        self.assertEqual([order[0][0] for order in orders],
                         ["fixed_n1", "fixed_n2", "fixed_n4", "fixed_n8", "adaptive"])
        self.assertTrue(all(sorted(order) == sorted(specs) for order in orders))
        self.assertEqual(benchmark.rotated_policy_specs(specs, 1, 1), orders[1])


if __name__ == "__main__":
    unittest.main()
