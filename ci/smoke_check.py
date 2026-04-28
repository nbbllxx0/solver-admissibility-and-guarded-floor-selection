from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    targets = [
        ROOT / "src" / "gpu_fem",
        ROOT / "experiments" / "phase5",
    ]
    failures: list[tuple[Path, Exception]] = []
    for base in targets:
        for path in sorted(base.glob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except Exception as exc:  # pragma: no cover - diagnostic script
                failures.append((path, exc))

    if failures:
        for path, exc in failures:
            print(f"FAILED {path.relative_to(ROOT)}: {exc}")
        raise SystemExit(1)

    print("OK: all release Python files compile.")


if __name__ == "__main__":
    main()
