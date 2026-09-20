import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("check_checkpoint", Path(__file__).resolve().parents[1] / "tools/check_checkpoint.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CheckpointIntegrityTests(unittest.TestCase):
    def test_valid_crc_does_not_claim_model_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("archive/data.pkl", b"not an executable pickle")
                archive.writestr("archive/data/0", b"tensor-placeholder")
            result = module.inspect_checkpoint(path)
            self.assertEqual(result["status"], "crc_passed_model_unverified")
            self.assertEqual(result["records_checked"], 2)

    def test_modified_storage_fails_crc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("archive/data.pkl", b"metadata")
                archive.writestr("archive/data/0", b"unique-tensor-bytes")
            data = path.read_bytes().replace(b"unique-tensor-bytes", b"Unique-tensor-bytes")
            path.write_bytes(data)
            result = module.inspect_checkpoint(path)
            self.assertEqual(result["status"], "integrity_failed")
            self.assertEqual(result["errors"][0]["entry"], "archive/data/0")

    def test_ordinary_zip_is_not_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordinary.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("hello.txt", "hello")
            self.assertEqual(module.inspect_checkpoint(path)["status"], "integrity_failed")

    def test_legacy_format_is_not_misreported_as_corrupt_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            path.write_bytes(b"not a ZIP archive")
            self.assertEqual(module.inspect_checkpoint(path)["status"], "unsupported_or_unreadable")


if __name__ == "__main__":
    unittest.main()
