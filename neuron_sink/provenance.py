"""Shared provenance, hardware-gate, and output-contract helpers.

Introduced for Task 4. Tasks 2 and 3 keep their own inline provenance writers so their
already-signed-off scripts stay byte-stable; the one thing they now import from here is
:func:`require_registered_gpu`, so the registered-hardware list has a single definition.

Schemas follow ``docs/05_METRICS_AND_SCHEMAS.md``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SINK_REPRO = ROOT / "upstream" / "sink-repro"
SINK_KD = ROOT / "upstream" / "sink-kd"

EXPECTED_SINK_REPRO_COMMIT = "9ab67e914464b13863b67527d8ea14068ee9ff10"
EXPECTED_SINK_KD_COMMIT = "db114c9c5eb6ffc5de13e444c783408ea7401c62"

#: Registered hardware, by role. ``dev`` lists both development GPUs because amendment
#: A001 (``docs/AMENDMENTS.md``) added the RTX 2060 12 GB alongside the RTX 2060 SUPER
#: that produced the Task-2/Task-3 results. ``configs/hardware_profiles.yaml`` and
#: ``docs/04_HARDWARE_RUNBOOK.md`` document these; this tuple is what code enforces.
REGISTERED_GPUS: Mapping[str, tuple[str, ...]] = {
    "dev": ("NVIDIA GeForce RTX 2060", "NVIDIA GeForce RTX 2060 SUPER"),
    "full": ("NVIDIA GeForce RTX 4080 SUPER",),
}

PACKAGE_VERSION_NAMES = ("torch", "transformers", "nnsight", "datasets", "numpy", "pandas")


class ProvenanceError(RuntimeError):
    """Raised when the runtime does not match the registered experiment contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8"
    ).strip()


def require_pinned_submodules() -> dict[str, str]:
    """Verify both upstream submodules are at their pinned commits and unmodified."""

    commits = {
        "sink_repro_commit": git("rev-parse", "HEAD", cwd=SINK_REPRO),
        "sink_kd_commit": git("rev-parse", "HEAD", cwd=SINK_KD),
    }
    expected = {
        "sink_repro_commit": EXPECTED_SINK_REPRO_COMMIT,
        "sink_kd_commit": EXPECTED_SINK_KD_COMMIT,
    }
    for key, value in commits.items():
        if value != expected[key]:
            raise ProvenanceError(f"{key} is {value}; expected {expected[key]}")
    for name, path in (("sink-repro", SINK_REPRO), ("sink-kd", SINK_KD)):
        if git("status", "--porcelain", cwd=path):
            raise ProvenanceError(f"upstream/{name} is modified; refusing to run")
    return commits


def submodules_are_clean() -> bool:
    return not (
        git("status", "--porcelain", cwd=SINK_REPRO)
        or git("status", "--porcelain", cwd=SINK_KD)
    )


def require_registered_gpu(profile: str) -> tuple[torch.device, str, int]:
    """Return ``(device, gpu_name, total_vram_bytes)`` for a registered hardware role."""

    if profile not in REGISTERED_GPUS:
        raise ProvenanceError(
            f"Unknown hardware profile {profile!r}; registered roles are "
            f"{sorted(REGISTERED_GPUS)}"
        )
    if not torch.cuda.is_available():
        raise ProvenanceError(
            f"The {profile!r} hardware profile requires CUDA to be usable by PyTorch"
        )
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    allowed = REGISTERED_GPUS[profile]
    if gpu_name not in allowed:
        raise ProvenanceError(
            f"GPU {gpu_name!r} is not registered for the {profile!r} profile. "
            f"Registered: {list(allowed)}. Changing hardware requires a documented "
            "amendment in docs/AMENDMENTS.md."
        )
    total_vram = int(torch.cuda.get_device_properties(device).total_memory)
    return device, gpu_name, total_vram


def package_versions(names: Iterable[str] = PACKAGE_VERSION_NAMES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def prepare_output_dir(path: Path) -> Path:
    """Create an empty run directory. Completed runs are append-only and never reused."""

    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Results are append-only."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(item: Any) -> Any:
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, Path):
        return str(item)
    raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Key-order-independent JSON text, so a hash cannot depend on dict ordering."""

    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=_json_default,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                   default=_json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class ProvenanceRecorder:
    """Collect the ``provenance.json`` fields required by docs/05_METRICS_AND_SCHEMAS.md."""

    device: torch.device | None = None
    gpu_name: str = "cpu"
    started_at: str = ""
    _wall_start: float = 0.0

    def __post_init__(self) -> None:
        self.started_at = utc_now()
        self._wall_start = time.perf_counter()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def finish(self, *, repo_commit: str, submodule_commits: Mapping[str, str]) -> dict:
        peak_allocated = 0
        peak_reserved = 0
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        wall_seconds = time.perf_counter() - self._wall_start
        return {
            "repo_commit": repo_commit,
            **dict(submodule_commits),
            "python": platform.python_version(),
            **package_versions(),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": self.gpu_name,
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "runtime_seconds": wall_seconds,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
        }
