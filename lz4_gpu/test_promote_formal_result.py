#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import promote_formal_result as promote


RUN_ID = "heterolz-admission-20260719T120000000000Z"


class PromoteFormalResultTest(unittest.TestCase):
    def make_auditor(self, root: Path, status: str = "complete") -> tuple[Path, str]:
        path = root / "auditor.py"
        exit_code = 0 if status == "complete" else 1
        path.write_text(
            "import json, sys\n"
            f"print(json.dumps({{'status': {status!r}, 'errors': [] if {status!r} == 'complete' else ['bad']}}))\n"
            f"sys.exit({exit_code})\n",
            encoding="utf-8",
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path, digest

    def make_archive(self, root: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> Path:
        path = root / "result.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for info, data in members:
                archive.addfile(info, io.BytesIO(data) if data is not None else None)
        return path

    def valid_members(self) -> list[tuple[tarfile.TarInfo, bytes | None]]:
        directory = tarfile.TarInfo(RUN_ID)
        directory.type = tarfile.DIRTYPE
        payload = json.dumps({"run_id": RUN_ID}).encode()
        evidence = tarfile.TarInfo(f"{RUN_ID}/run_manifest.json")
        evidence.size = len(payload)
        return [(directory, None), (evidence, payload)]

    def test_promotes_only_after_complete_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.make_archive(root, self.valid_members())
            auditor, digest = self.make_auditor(root)
            results = root / "runs"
            final, audit = promote.promote_local_archive(
                archive, RUN_ID, results, auditor, expected_auditor_sha256=digest
            )
            self.assertEqual(audit["status"], "complete")
            self.assertTrue((final / "run_manifest.json").is_file())
            self.assertEqual([path.name for path in results.iterdir()], [RUN_ID])
            with self.assertRaises(FileExistsError):
                promote.promote_local_archive(
                    archive, RUN_ID, results, auditor, expected_auditor_sha256=digest
                )

    def test_rejects_path_traversal_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            info = tarfile.TarInfo(f"{RUN_ID}/../escape")
            info.size = 1
            archive = self.make_archive(root, [(info, b"x")])
            auditor, digest = self.make_auditor(root)
            results = root / "runs"
            with self.assertRaises(ValueError):
                promote.promote_local_archive(
                    archive, RUN_ID, results, auditor, expected_auditor_sha256=digest
                )
            self.assertEqual(list(results.iterdir()), [])
            self.assertFalse((root / "escape").exists())

    def test_rejects_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = tarfile.TarInfo(RUN_ID)
            directory.type = tarfile.DIRTYPE
            link = tarfile.TarInfo(f"{RUN_ID}/run_manifest.json")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive = self.make_archive(root, [(directory, None), (link, None)])
            auditor, digest = self.make_auditor(root)
            with self.assertRaises(ValueError):
                promote.promote_local_archive(
                    archive, RUN_ID, root / "runs", auditor,
                    expected_auditor_sha256=digest,
                )

    def test_incomplete_audit_leaves_no_result_or_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.make_archive(root, self.valid_members())
            auditor, digest = self.make_auditor(root, "incomplete")
            results = root / "runs"
            with self.assertRaises(RuntimeError):
                promote.promote_local_archive(
                    archive, RUN_ID, results, auditor, expected_auditor_sha256=digest
                )
            self.assertEqual(list(results.iterdir()), [])

    def test_validates_identifiers_roots_and_checksums(self) -> None:
        self.assertEqual(promote.validate_run_id(RUN_ID), RUN_ID)
        self.assertEqual(promote.validate_remote("root@192.168.2.225"), "root@192.168.2.225")
        self.assertEqual(promote.validate_remote_root("/root/heterolz-formal-results"),
                         "/root/heterolz-formal-results")
        self.assertEqual(
            promote.validate_formal_endpoint(
                "root@192.168.2.225", "/root/heterolz-formal-results"
            ),
            ("root@192.168.2.225", "/root/heterolz-formal-results"),
        )
        digest = "a" * 64
        self.assertEqual(promote.parse_sha256sum(f"{digest}  /tmp/a.tar.gz\n"), digest)
        for value in ("../run", "heterolz-admission-latest", "x;rm"):
            with self.assertRaises(ValueError):
                promote.validate_run_id(value)
        for value in ("relative/path", "/", "/root/../tmp", "/root/a path"):
            with self.assertRaises(ValueError):
                promote.validate_remote_root(value)
        for remote, root in (
            ("root@192.168.2.105", "/root/heterolz-formal-results"),
            ("root@192.168.2.225", "/tmp/results"),
        ):
            with self.assertRaises(ValueError):
                promote.validate_formal_endpoint(remote, root)
        with self.assertRaises(RuntimeError):
            promote.parse_sha256sum("not-a-digest")
        with tempfile.TemporaryDirectory() as temp:
            other_auditor = Path(temp) / "auditor.py"
            other_auditor.write_text("print('x')\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                promote.validate_formal_auditor(other_auditor)

    def test_result_root_hygiene_allows_only_readme_and_formal_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("contract\n", encoding="utf-8")
            (root / RUN_ID).mkdir()
            promote.validate_result_root_contents(root, allow_readme=True)
            manual = root / "manual.tmp"
            manual.write_text("leftover\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                promote.validate_result_root_contents(root, allow_readme=True)


if __name__ == "__main__":
    unittest.main()
