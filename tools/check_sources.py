"""仅检查迁移文件完整性，不代表模型可运行。"""
import ast
import hashlib
import json
from pathlib import Path


def verify(root):
    errors = []
    records = json.loads((root / "docs/source_manifest.json").read_text())
    for record in records:
        original = root / "artifacts/original_source" / record["source"]
        if original.exists() and hashlib.sha256(original.read_bytes()).hexdigest() != record["source_sha256"]:
            errors.append(f"原始材料哈希变化：{record['source']}")
        if "destination" not in record:
            continue
        path = root / record["destination"]
        try:
            data = path.read_bytes()
            ast.parse(data)
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                errors.append(f"已审查版本哈希已变化：{record['destination']}")
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"{record['destination']}: {exc}")
    return errors


if __name__ == "__main__":
    errors = verify(Path(__file__).resolve().parents[1])
    print("\n".join(errors) if errors else "源码语法及已审查版本哈希通过；运行验证请执行 tests。")
    raise SystemExit(bool(errors))
