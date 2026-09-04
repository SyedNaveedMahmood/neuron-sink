"""Import pinned upstream ``common/`` modules without letting the two snapshots collide.

Both submodules ship a ``common/`` directory of *top-level* modules (no ``__init__.py``)
whose names collide with each other and with this project: ``datasets_loader``,
``intervention_analysis_legacy``, ``nnsight_engine``, ``corpus_providers``, and --
critically -- ``provenance``. They also mix import styles: sink-kd's ``corpus_providers``
uses a ``try: from . import x / except ImportError: import x`` fallback, while
sink-repro's ``intervention_analysis_legacy`` uses a plain absolute
``from datasets_loader import ...``. Only putting the snapshot's directory on ``sys.path``
satisfies both.

Doing that permanently -- as ``scripts/run_gpt2_sink_parity.py`` does for sink-repro --
would mean the first snapshot imported wins for the rest of the process, so sink-kd's
``corpus_providers`` could silently be handed sink-repro's ``datasets_loader``, and
``neuron_sink.provenance`` could be shadowed. This module therefore scopes the ``sys.path``
entry to the import itself, then harvests every module that was loaded out of that
directory into a private per-snapshot cache and removes it from ``sys.modules``. A later
import from the same snapshot reinstalls its own cache first, so intra-snapshot imports
resolve within the correct pinned tree.

Nothing here edits or copies upstream code; it only controls how it is imported.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from pathlib import Path

from .provenance import SINK_KD, SINK_REPRO


_COMMON_DIRS = {
    "sink_repro": SINK_REPRO / "common",
    "sink_kd": SINK_KD / "common",
}

#: Modules harvested out of each snapshot, keyed by snapshot then module name.
_loaded: dict[str, dict[str, types.ModuleType]] = {}


class UpstreamNotAvailableError(ImportError):
    """Raised when a pinned submodule is not checked out."""


def common_dir(which: str) -> Path:
    try:
        return _COMMON_DIRS[which]
    except KeyError:
        raise ValueError(
            f"Unknown upstream {which!r}; expected one of {sorted(_COMMON_DIRS)}"
        ) from None


def is_available(which: str) -> bool:
    directory = common_dir(which)
    return directory.is_dir() and any(directory.glob("*.py"))


@contextlib.contextmanager
def _snapshot_namespace(which: str):
    """Expose one snapshot's ``common/`` for the duration of an import, then withdraw it."""

    directory = common_dir(which)
    if not is_available(which):
        raise UpstreamNotAvailableError(
            f"{directory} is missing or empty. Run: "
            "GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive"
        )
    resolved = directory.resolve()
    cache = _loaded.setdefault(which, {})

    # Reinstall this snapshot's previously harvested modules so its internal imports
    # resolve to its own files rather than re-importing or hitting the other snapshot.
    shadowed: dict[str, types.ModuleType | None] = {}
    for name, module in cache.items():
        shadowed[name] = sys.modules.get(name)
        sys.modules[name] = module

    path_entry = str(directory)
    sys.path.insert(0, path_entry)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(path_entry)
        # Harvest everything that came from this snapshot's directory.
        for name in list(sys.modules):
            module = sys.modules.get(name)
            file = getattr(module, "__file__", None)
            if not file:
                continue
            try:
                if Path(file).resolve().parent != resolved:
                    continue
            except (OSError, ValueError):
                continue
            cache[name] = sys.modules.pop(name)
        # Restore anything this context shadowed but did not harvest.
        for name, previous in shadowed.items():
            if name in sys.modules:
                continue
            if previous is not None:
                sys.modules[name] = previous


def upstream_module(which: str, module_name: str) -> types.ModuleType:
    """Import ``module_name`` from the pinned ``which`` snapshot's ``common/``."""

    cache = _loaded.setdefault(which, {})
    if module_name in cache:
        return cache[module_name]
    with _snapshot_namespace(which):
        importlib.import_module(module_name)
    try:
        return _loaded[which][module_name]
    except KeyError:  # pragma: no cover - defensive
        raise ImportError(
            f"{module_name!r} did not load from {common_dir(which)}"
        ) from None


def sink_repro_module(module_name: str) -> types.ModuleType:
    return upstream_module("sink_repro", module_name)


def sink_kd_module(module_name: str) -> types.ModuleType:
    return upstream_module("sink_kd", module_name)
