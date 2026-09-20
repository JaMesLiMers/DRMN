"""逐项检查 PyTorch ZIP 权重的 CRC；不反序列化或执行权重。"""
import argparse
import json
from pathlib import Path
import zipfile


def inspect_checkpoint(path):
    result = {"path": str(path), "records_checked": 0, "errors": [], "status": "unverified"}
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata_present = any(n == "data.pkl" or n.endswith("/data.pkl") for n in names)
            tensor_present = any("/data/" in n or n.startswith("data/") for n in names)
            if not metadata_present or not tensor_present:
                result["errors"].append({"entry": None, "reason": "不是包含 data.pkl 和张量存储的权重 ZIP"})
            for info in archive.infolist():
                if info.is_dir():
                    continue
                result["records_checked"] += 1
                try:
                    with archive.open(info) as stream:
                        while stream.read(4 * 1024 * 1024):
                            pass
                except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as error:
                    result["errors"].append({"entry": info.filename, "reason": str(error)})
        result["status"] = "integrity_failed" if result["errors"] else "crc_passed_model_unverified"
    except (OSError, zipfile.BadZipFile) as error:
        result["errors"].append({"entry": None, "reason": str(error)})
        result["status"] = "unsupported_or_unreadable"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_checkpoint(args.checkpoint)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["status"] == "crc_passed_model_unverified" else 1)
