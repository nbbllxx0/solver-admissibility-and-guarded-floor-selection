"""Source-level release check.

Parses every released Python file and verifies that the files every GPU driver
depends on are present. It intentionally does not import CUDA modules or run
GPU solves, so it is safe to run in CI or on a machine without an NVIDIA GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    ROOT / "src" / "gpu_fem",
    ROOT / "experiments" / "phase5",
    ROOT / "experiments" / "paper4",
]

# A release missing one of these looks complete but cannot run anything.
REQUIRED = [
    ROOT / "experiments" / "paper4" / "run_experiments_e1_e10.py",
    ROOT / "src" / "gpu_fem" / "multigrid_v4.py",
    ROOT / "src" / "gpu_fem" / "presets.py",
    ROOT / "src" / "gpu_fem" / "bc_generator.py",
    ROOT / "experiments" / "phase5" / "run_gmg_floor_detector_prospective.py",
    ROOT / "experiments" / "phase5" / "run_gmg_floor_detector_density_field.py",
    ROOT / "environment.yml",
]


def main() -> None:
    failures: list[tuple[Path, Exception]] = []
    checked = 0
    for base in TARGETS:
        if not base.is_dir():
            failures.append((base, FileNotFoundError("missing directory")))
            continue
        for path in sorted(base.glob("*.py")):
            checked += 1
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except Exception as exc:  # pragma: no cover - diagnostic script
                failures.append((path, exc))

    for path in REQUIRED:
        if not path.exists():
            failures.append((path, FileNotFoundError("required file missing from release")))

    if failures:
        for path, exc in failures:
            try:
                rel = path.relative_to(ROOT)
            except ValueError:
                rel = path
            print(f"FAILED {rel}: {exc}")
        raise SystemExit(1)

    print(f"OK: {checked} release Python files compile; all required files present.")


if __name__ == "__main__":
    main()
