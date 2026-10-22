"""One-time migration authorized for the develop branch; preserves OCR data and report."""
from hashlib import sha256
from pathlib import Path
import json
import shutil
import subprocess


def migrate():
    root = Path(__file__).resolve().parents[1]
    branch = subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip()
    if branch != "develop":
        raise RuntimeError("Legacy cleanup is restricted to develop")
    if (root / "docs" / "migration-receipt.json").exists():
        raise RuntimeError("Migration already completed; refusing to repeat destructive cleanup")
    dataset = (root / "CNN" / "CNN_letter_Dataset").resolve(strict=True)
    def inventory():
        return sorted((p.relative_to(dataset).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                      for p in dataset.rglob("*") if p.is_file())
    before = inventory()
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    legacy_report = docs / "legacy-report.md"
    if not legacy_report.exists():
        content = (root / "README.MD").read_text(encoding="utf-8")
        for folder in ("CNN", "CNN_threshold", "KNN", "SVM"):
            content = content.replace(f"./{folder}/run/", f"./legacy-evidence/{folder}/")
        for relative in ("img/", "3.jpg", "carlong_0004.png"):
            content = content.replace(f"./{relative}", f"../{relative}")
        legacy_report.write_text(
            "> Báo cáo lịch sử trước migration ML cổ điển; metrics dưới đây thuộc model cũ đã xóa.\n\n" + content,
            encoding="utf-8")
    removed, evidence = [], []
    for folder in ("CNN", "CNN_threshold", "KNN", "SVM"):
        old = (root / folder).resolve()
        if not old.is_relative_to(root) or old == root:
            raise RuntimeError(f"Unsafe legacy source: {old}")
        run = old / "run"
        if run.is_dir():
            target = (docs / "legacy-evidence" / folder).resolve()
            if not target.is_relative_to(root) or target.exists():
                raise RuntimeError(f"Unexpected evidence target: {target}")
            target.parent.mkdir(exist_ok=True)
            evidence.append(str(target.relative_to(root)))
            shutil.move(str(run), str(target))
        targets = list(old.iterdir()) if folder == "CNN" else [old]
        for candidate in targets:
            target = candidate.resolve()
            if target == dataset:
                continue
            if not target.is_relative_to(root) or target == root or target == dataset or dataset.is_relative_to(target):
                raise RuntimeError(f"Unsafe deletion target: {target}")
            removed.append(str(target.relative_to(root)))
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
    after = inventory()
    if before != after:
        raise RuntimeError("Dataset inventory changed during migration")
    report = dict(branch=branch, dataset=str(dataset.relative_to(root)), dataset_files=len(after),
                  dataset_inventory_sha256=sha256(json.dumps(after).encode()).hexdigest(),
                  check="relative paths, byte lengths and nanosecond mtimes identical before/after",
                  removed=removed, evidence_preserved=evidence)
    (docs / "migration-receipt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(migrate(), indent=2))
