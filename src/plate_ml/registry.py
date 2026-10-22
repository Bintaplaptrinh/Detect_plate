from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from uuid import uuid4
import json
import platform
import re
import subprocess
import joblib


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict:
    packages = {}
    for name in ("numpy", "opencv-python", "scikit-learn", "scipy", "joblib", "vn-plate-ml"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages)


def code_info(root: str | Path) -> dict:
    root = Path(root).resolve()
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
        return result.stdout.decode("utf-8", errors="replace").strip()
    try:
        revision, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, None
    digest = sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    return dict(git_revision=revision, dirty=dirty, source_sha256=digest.hexdigest())


def source_snapshot(root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    return {path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((root / "src").rglob("*.py"))}


def _safe_name(name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError("Invalid artifact name")
    return name


def _write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


class ModelRegistry:
    """Immutable Registration/Registry/<name>/vX releases and run checkpoints.

    Directory creation reserves a unique version even across concurrent writers.
    metadata.json is the commit marker. Incomplete versions are never listed.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def register(self, name: str, bundle, metadata: dict, files: dict | None = None) -> dict:
        name = _safe_name(name)
        # Validate JSON before reserving a version.
        json.dumps(metadata, allow_nan=False)
        base = self.root / "Registry" / name
        base.mkdir(parents=True, exist_ok=True)
        version_number = 1
        while True:
            path = base / f"v{version_number}"
            try:
                path.mkdir()
                break
            except FileExistsError:
                version_number += 1
        model_path = path / "model.joblib"
        joblib.dump(bundle, model_path, compress=3)
        checksums = {"model.joblib": file_sha256(model_path)}
        for key, value in (files or {}).items():
            filename = _safe_name(key) + ".json"
            if filename == "metadata.json":
                raise ValueError("Reserved artifact filename")
            _write_json(path / filename, value)
            checksums[filename] = file_sha256(path / filename)
        record = dict(schema_version=1, name=name, version=f"v{version_number}",
                      created_at=datetime.now(timezone.utc).isoformat(), environment=environment_info(),
                      artifact_sha256=checksums, metadata=metadata)
        _write_json(path / "metadata.json", record)
        return dict(path=str(path), **record)

    def list_versions(self, name: str) -> list[dict]:
        base = self.root / "Registry" / _safe_name(name)
        paths = [p for p in base.glob("v*") if re.fullmatch(r"v[1-9][0-9]*", p.name)]
        records = []
        for path in sorted(paths, key=lambda p: int(p.name[1:])):
            if (path / "metadata.json").is_file():
                records.append(dict(path=str(path), **json.loads((path / "metadata.json").read_text(encoding="utf-8"))))
        return records

    def load(self, name: str, version_id: str, *, trusted: bool = False):
        if not trusted:
            raise ValueError("joblib can execute code; load only a locally trusted release with trusted=True")
        if not re.fullmatch(r"v[1-9][0-9]*", version_id):
            raise ValueError("Invalid version ID")
        path = self.root / "Registry" / _safe_name(name) / version_id
        record = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        for filename, expected in record["artifact_sha256"].items():
            if Path(filename).name != filename or not re.fullmatch(r"[a-zA-Z0-9_.-]+", filename):
                raise ValueError("Unsafe artifact filename")
            if file_sha256(path / filename) != expected:
                raise ValueError(f"Artifact integrity check failed: {filename}")
        return joblib.load(path / "model.joblib"), record

    def checkpoint_writer(self, run_id: str, context: dict | None = None):
        json.dumps(context or {}, allow_nan=False)
        run_path = self.root / "Checkpoints" / _safe_name(run_id)
        run_path.mkdir(parents=True, exist_ok=False)
        counts = {}

        def save(name, model, metadata):
            name = _safe_name(name)
            counts[name] = counts.get(name, 0) + 1
            path = run_path / f"{name}-{counts[name]:03d}.joblib"
            joblib.dump(model, path, compress=3)
            _write_json(path.with_suffix(".json"), dict(created_at=datetime.now(timezone.utc).isoformat(),
                        sha256=file_sha256(path), environment=environment_info(), context=context or {}, **metadata))
        return save
