from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path(r"C:\Users\YSL0\anaconda3\envs\PY\python.exe")


@dataclass
class ChunkResult:
    name: str
    command: list[str]
    started_at: float
    ended_at: float | None
    returncode: int | None
    status: str
    output_lines: int
    idle_seconds: float
    elapsed_seconds: float


def _line_reader(stream, out_queue: queue.Queue[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            out_queue.put(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _write_status(path: Path, results: list[ChunkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(item) for item in results], indent=2),
        encoding="utf-8",
    )


def _run_chunk(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    observer_log,
    hard_timeout_s: float,
    idle_timeout_s: float,
) -> ChunkResult:
    started = time.time()
    observer_log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {name}\n")
    observer_log.write("COMMAND " + " ".join(command) + "\n")
    observer_log.flush()

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    q: queue.Queue[str] = queue.Queue()
    assert process.stdout is not None
    reader = threading.Thread(target=_line_reader, args=(process.stdout, q), daemon=True)
    reader.start()

    output_lines = 0
    last_output = time.time()
    status = "running"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as chunk_log:
        chunk_log.write("COMMAND " + " ".join(command) + "\n")
        while True:
            now = time.time()
            while True:
                try:
                    line = q.get_nowait()
                except queue.Empty:
                    break
                output_lines += 1
                last_output = now
                chunk_log.write(line)
                chunk_log.flush()
                observer_log.write(f"[{name}] {line}")
                observer_log.flush()

            returncode = process.poll()
            if returncode is not None:
                status = "completed" if returncode == 0 else "failed"
                break

            elapsed = now - started
            idle = now - last_output
            if elapsed > hard_timeout_s:
                status = "hard_timeout"
                process.kill()
                break
            if idle > idle_timeout_s:
                status = "idle_timeout"
                process.kill()
                break

            if int(elapsed) % 60 == 0:
                observer_log.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HEARTBEAT {name}: "
                    f"elapsed={elapsed:.0f}s idle={idle:.0f}s lines={output_lines}\n"
                )
                observer_log.flush()
                time.sleep(1.0)
            else:
                time.sleep(2.0)

        reader.join(timeout=5)
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                break
            output_lines += 1
            chunk_log.write(line)
            observer_log.write(f"[{name}] {line}")

    ended = time.time()
    idle = ended - last_output
    elapsed = ended - started
    rc = process.poll()
    observer_log.write(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END {name}: "
        f"status={status} returncode={rc} elapsed={elapsed:.1f}s idle={idle:.1f}s "
        f"lines={output_lines}\n"
    )
    observer_log.flush()
    return ChunkResult(
        name=name,
        command=command,
        started_at=started,
        ended_at=ended,
        returncode=rc,
        status=status,
        output_lines=output_lines,
        idle_seconds=idle,
        elapsed_seconds=elapsed,
    )


def _prospective_command(
    python: Path,
    *,
    preset: str,
    seeds: str,
    probabilities: str,
    out_dir: Path,
    baseline_rho_min: str = "1e-12",
    penal: str = "4.5",
    stack_variant: str = "canonical",
    high_threshold: str = "1e99",
    plateau_threshold: str = "1e99",
    severity_jump_r50_threshold: str | None = None,
    severity_jump_rho_min: str = "1e-2",
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        str(python),
        "experiments/phase5/run_gmg_floor_detector_prospective.py",
        "--preset",
        preset,
        "--seeds",
        seeds,
        "--probabilities",
        probabilities,
        "--baseline-rho-min",
        baseline_rho_min,
        "--penal",
        penal,
        "--stack-variant",
        stack_variant,
        "--raised-rho-mins",
        "1e-3,1e-2",
        "--high-residual-threshold",
        high_threshold,
        "--plateau-residual-threshold",
        plateau_threshold,
        "--out-dir",
        str(out_dir),
    ]
    if severity_jump_r50_threshold is not None:
        command.extend(
            [
                "--severity-jump-r50-threshold",
                severity_jump_r50_threshold,
                "--severity-jump-rho-min",
                severity_jump_rho_min,
            ]
        )
    if extra_args:
        command.extend(extra_args)
    return command


def _full_label_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    chunks: list[tuple[str, list[str], Path, Path]] = []
    cantilever_probs = "0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35"
    bridge_probs = "0.10,0.15,0.20,0.25,0.30,0.35"
    for seed in [47, 53, 59, 61, 67, 71]:
        out_dir = ROOT / "experiments" / "phase5" / "results" / f"heldout_full_true_labels_cantilever_s{seed}"
        chunks.append(
            (
                f"full_label_cantilever_s{seed}",
                _prospective_command(
                    python,
                    preset="cantilever_gpu_medium",
                    seeds=str(seed),
                    probabilities=cantilever_probs,
                    out_dir=out_dir,
                ),
                out_dir / "observer_chunk.log",
                out_dir / "prospective_summary.csv",
            )
        )
    for seed in [41, 43, 47, 53, 59]:
        out_dir = ROOT / "experiments" / "phase5" / "results" / f"heldout_full_true_labels_bridge_s{seed}"
        chunks.append(
            (
                f"full_label_bridge_s{seed}",
                _prospective_command(
                    python,
                    preset="bridge_gpu_medium",
                    seeds=str(seed),
                    probabilities=bridge_probs,
                    out_dir=out_dir,
                ),
                out_dir / "observer_chunk.log",
                out_dir / "prospective_summary.csv",
            )
        )
    return chunks


def _write_fixed_floor_manifest(
    *,
    manifest_path: Path,
    preset: str,
    seed: int,
    probabilities: list[float],
    floors: str,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "case_type",
        "preset",
        "seed",
        "solid_probability",
        "density_path",
        "penal",
        "floors",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for probability in probabilities:
            writer.writerow(
                {
                    "case_id": f"{preset}_s{seed}_q{probability:g}",
                    "case_type": "random",
                    "preset": preset,
                    "seed": seed,
                    "solid_probability": f"{probability:g}",
                    "density_path": "",
                    "penal": "4.5",
                    "floors": floors,
                }
            )


def _fixed_floor_command(
    python: Path,
    *,
    manifest: Path,
    out_dir: Path,
) -> list[str]:
    return [
        str(python),
        "experiments/phase5/run_gmg_fixed_floor_controls.py",
        "--manifest",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--restart",
        "300",
        "--maxiter",
        "300",
        "--tol",
        "1e-6",
    ]


def _fixed_floor_baseline_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    chunks: list[tuple[str, list[str], Path, Path]] = []
    root = ROOT / "experiments" / "phase5" / "results" / "policy_fixed_floor_baselines"
    manifest_root = root / "manifests"
    floors = "1e-3;1e-2"
    cases = [
        ("cantilever_gpu_medium", [41, 43, 47, 53, 59, 61, 67, 71], [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]),
        ("bridge_gpu_medium", [41, 43, 47, 53, 59], [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]),
    ]
    for preset, seeds, probabilities in cases:
        short = "cantilever" if preset.startswith("cantilever") else "bridge"
        for seed in seeds:
            out_dir = root / f"{short}_s{seed}"
            manifest = manifest_root / f"{short}_s{seed}.csv"
            _write_fixed_floor_manifest(
                manifest_path=manifest,
                preset=preset,
                seed=seed,
                probabilities=probabilities,
                floors=floors,
            )
            chunks.append(
                (
                    f"fixed_floor_{short}_s{seed}",
                    _fixed_floor_command(python, manifest=manifest, out_dir=out_dir),
                    out_dir / "observer_chunk.log",
                    out_dir / "fixed_floor_control_summary.csv",
                )
            )
    return chunks


def _severity_jump_baseline_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    chunks: list[tuple[str, list[str], Path, Path]] = []
    root = ROOT / "experiments" / "phase5" / "results" / "policy_severity_jump_baselines"
    cases = [
        (
            "cantilever_gpu_medium",
            "cantilever",
            [41, 43, 47, 53, 59, 61, 67, 71],
            "0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35",
        ),
        (
            "bridge_gpu_medium",
            "bridge",
            [41, 43, 47, 53, 59],
            "0.10,0.15,0.20,0.25,0.30,0.35",
        ),
    ]
    for preset, short, seeds, probabilities in cases:
        for seed in seeds:
            out_dir = root / f"{short}_s{seed}"
            chunks.append(
                (
                    f"severity_jump_{short}_s{seed}",
                    _prospective_command(
                        python,
                        preset=preset,
                        seeds=str(seed),
                        probabilities=probabilities,
                        out_dir=out_dir,
                        high_threshold="1e-2",
                        plateau_threshold="1e-4",
                        severity_jump_r50_threshold="5e-1",
                        severity_jump_rho_min="1e-2",
                    ),
                    out_dir / "observer_chunk.log",
                    out_dir / "prospective_summary.csv",
                )
            )
    return chunks


def _sensitivity_perturbation_command(
    python: Path,
    *,
    seed: int,
    out_dir: Path,
) -> list[str]:
    return [
        str(python),
        "experiments/phase5/analyze_gmg_sensitivity_perturbation.py",
        "--case-filter-seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]


def _sensitivity_perturbation_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    chunks: list[tuple[str, list[str], Path, Path]] = []
    root = ROOT / "experiments" / "phase5" / "results" / "heldout_true_keep_sensitivity_perturbation"
    for seed in [41, 43, 47, 53, 59, 61, 67]:
        out_dir = root / f"cantilever_s{seed}"
        chunks.append(
            (
                f"sensitivity_cantilever_s{seed}",
                _sensitivity_perturbation_command(python, seed=seed, out_dir=out_dir),
                out_dir / "observer_chunk.log",
                out_dir / "sensitivity_perturbation_summary.csv",
            )
        )
    return chunks


def _sensitivity_case_chunks(
    python: Path,
    *,
    root_name: str,
    values: list[tuple[str, dict[str, str]]],
) -> list[tuple[str, list[str], Path, Path]]:
    chunks: list[tuple[str, list[str], Path, Path]] = []
    root = ROOT / "experiments" / "phase5" / "results" / root_name
    cases = [
        ("cantilever_gpu_medium", "cantilever", [41, 43], "0.10,0.20,0.35"),
        ("bridge_gpu_medium", "bridge", [41, 43], "0.10,0.20,0.35"),
    ]
    for value_label, overrides in values:
        for preset, short, seeds, probabilities in cases:
            for seed in seeds:
                out_dir = root / value_label / f"{short}_s{seed}"
                chunks.append(
                    (
                        f"{root_name}_{value_label}_{short}_s{seed}",
                        _prospective_command(
                            python,
                            preset=preset,
                            seeds=str(seed),
                            probabilities=probabilities,
                            out_dir=out_dir,
                            **overrides,
                        ),
                        out_dir / "observer_chunk.log",
                        out_dir / "prospective_summary.csv",
                    )
                )
    return chunks


def _simp_exponent_sensitivity_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    return _sensitivity_case_chunks(
        python,
        root_name="simp_exponent_sensitivity",
        values=[
            ("p3p0", {"penal": "3.0"}),
            ("p4p0", {"penal": "4.0"}),
            ("p4p5", {"penal": "4.5"}),
        ],
    )


def _simp_exponent_policy_sensitivity_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    chunks = _sensitivity_case_chunks(
        python,
        root_name="simp_exponent_policy_sensitivity",
        values=[
            ("p3p0", {"penal": "3.0"}),
            ("p4p0", {"penal": "4.0"}),
            ("p4p5", {"penal": "4.5"}),
        ],
    )
    for _, command, _, _ in chunks:
        high_idx = command.index("--high-residual-threshold") + 1
        plateau_idx = command.index("--plateau-residual-threshold") + 1
        command[high_idx] = "1e-2"
        command[plateau_idx] = "1e-4"
    return chunks


def _original_floor_sensitivity_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    return _sensitivity_case_chunks(
        python,
        root_name="original_floor_sensitivity",
        values=[
            ("rho1em12", {"baseline_rho_min": "1e-12"}),
            ("rho1em9", {"baseline_rho_min": "1e-9"}),
            ("rho1em8", {"baseline_rho_min": "1e-8"}),
            ("rho1em6", {"baseline_rho_min": "1e-6"}),
        ],
    )


def _original_floor_policy_sensitivity_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    chunks = _sensitivity_case_chunks(
        python,
        root_name="original_floor_policy_sensitivity",
        values=[
            ("rho1em12", {"baseline_rho_min": "1e-12"}),
            ("rho1em9", {"baseline_rho_min": "1e-9"}),
            ("rho1em8", {"baseline_rho_min": "1e-8"}),
            ("rho1em6", {"baseline_rho_min": "1e-6"}),
        ],
    )
    for _, command, _, _ in chunks:
        high_idx = command.index("--high-residual-threshold") + 1
        plateau_idx = command.index("--plateau-residual-threshold") + 1
        command[high_idx] = "1e-2"
        command[plateau_idx] = "1e-4"
    return chunks


def _mechanism_ablation_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    variants: list[tuple[str, list[str]]] = [
        ("canonical", []),
        ("levels3", ["--n-levels", "3"]),
        ("jacobi_smoother", ["--smoother-type", "jacobi"]),
        ("w_cycle", ["--cycle-type", "w"]),
        (
            "no_root_correction",
            [
                "--coarse-correction-policy",
                "none",
                "--root-local-correction-mode",
                "none",
                "--inner-krylov-steps",
                "0",
            ],
        ),
        ("tol1em5", ["--tol", "1e-5"]),
        ("tol1em7", ["--tol", "1e-7"]),
    ]
    chunks: list[tuple[str, list[str], Path, Path]] = []
    root = ROOT / "experiments" / "phase5" / "results" / "mechanism_ablation"
    cases = [
        ("cantilever_gpu_medium", "cantilever", [43], "0.10,0.20,0.35"),
        ("bridge_gpu_medium", "bridge", [43], "0.10,0.20,0.35"),
    ]
    for variant, extra_args in variants:
        for preset, short, seeds, probabilities in cases:
            for seed in seeds:
                out_dir = root / variant / f"{short}_s{seed}"
                chunks.append(
                    (
                        f"mechanism_{variant}_{short}_s{seed}",
                        _prospective_command(
                            python,
                            preset=preset,
                            seeds=str(seed),
                            probabilities=probabilities,
                            out_dir=out_dir,
                            stack_variant=variant,
                            high_threshold="1e-2",
                            plateau_threshold="1e-4",
                            extra_args=extra_args,
                        ),
                        out_dir / "observer_chunk.log",
                        out_dir / "prospective_summary.csv",
                    )
                )
    return chunks


def _simp_floor_trajectory_chunks(python: Path) -> list[tuple[str, list[str], Path, Path]]:
    chunks: list[tuple[str, list[str], Path, Path]] = []
    root = ROOT / "experiments" / "phase5" / "results" / "simp_floor_trajectories"
    for preset, short in [
        ("cantilever_gpu_medium", "cantilever"),
        ("bridge_gpu_medium", "bridge"),
    ]:
        out_dir = root / short
        chunks.append(
            (
                f"simp_floor_trajectory_{short}",
                [
                    str(python),
                    "experiments/phase5/run_simp_floor_trajectory.py",
                    "--presets",
                    preset,
                    "--rho-mins",
                    "1e-12,1e-3,1e-2",
                    "--iters",
                    "40",
                    "--out-dir",
                    str(out_dir),
                ],
                out_dir / "observer_chunk.log",
                out_dir / "trajectory_summary.csv",
            )
        )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[
            "full_labels_remaining",
            "fixed_floor_baselines",
            "severity_jump_baselines",
            "sensitivity_perturbation",
            "simp_exponent_sensitivity",
            "original_floor_sensitivity",
            "simp_exponent_policy_sensitivity",
            "original_floor_policy_sensitivity",
            "mechanism_ablation",
            "simp_floor_trajectories",
            "review_extension_sweeps",
            "policy_sensitivity_sweeps",
        ],
        default="full_labels_remaining",
    )
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--hard-timeout-min", type=float, default=75.0)
    parser.add_argument("--idle-timeout-min", type=float, default=25.0)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means run all chunks")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()

    python = Path(args.python)
    if not python.exists():
        raise FileNotFoundError(python)

    run_root = ROOT / "experiments" / "phase5" / "results" / "observed_queues" / args.task
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    if args.task == "full_labels_remaining":
        chunks = _full_label_chunks(python)
    elif args.task == "fixed_floor_baselines":
        chunks = _fixed_floor_baseline_chunks(python)
    elif args.task == "severity_jump_baselines":
        chunks = _severity_jump_baseline_chunks(python)
    elif args.task == "sensitivity_perturbation":
        chunks = _sensitivity_perturbation_chunks(python)
    elif args.task == "simp_exponent_sensitivity":
        chunks = _simp_exponent_sensitivity_chunks(python)
    elif args.task == "original_floor_sensitivity":
        chunks = _original_floor_sensitivity_chunks(python)
    elif args.task == "simp_exponent_policy_sensitivity":
        chunks = _simp_exponent_policy_sensitivity_chunks(python)
    elif args.task == "original_floor_policy_sensitivity":
        chunks = _original_floor_policy_sensitivity_chunks(python)
    elif args.task == "mechanism_ablation":
        chunks = _mechanism_ablation_chunks(python)
    elif args.task == "simp_floor_trajectories":
        chunks = _simp_floor_trajectory_chunks(python)
    elif args.task == "review_extension_sweeps":
        chunks = (
            _simp_exponent_sensitivity_chunks(python)
            + _original_floor_sensitivity_chunks(python)
            + _mechanism_ablation_chunks(python)
        )
    elif args.task == "policy_sensitivity_sweeps":
        chunks = (
            _simp_exponent_policy_sensitivity_chunks(python)
            + _original_floor_policy_sensitivity_chunks(python)
        )
    else:
        raise ValueError(args.task)
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    results: list[ChunkResult] = []
    with (run_root / "observer.log").open("a", encoding="utf-8") as observer_log:
        observer_log.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] QUEUE START task={args.task} chunks={len(chunks)}\n"
        )
        observer_log.flush()
        for name, command, log_path, expected_output in chunks:
            if expected_output.exists() and not args.rerun_completed:
                now = time.time()
                result = ChunkResult(
                    name=name,
                    command=command,
                    started_at=now,
                    ended_at=now,
                    returncode=0,
                    status="skipped_completed",
                    output_lines=0,
                    idle_seconds=0.0,
                    elapsed_seconds=0.0,
                )
                results.append(result)
                _write_status(status_path, results)
                observer_log.write(f"SKIP {name}: found {expected_output}\n")
                observer_log.flush()
                continue
            result = _run_chunk(
                name=name,
                command=command,
                log_path=log_path,
                observer_log=observer_log,
                hard_timeout_s=args.hard_timeout_min * 60.0,
                idle_timeout_s=args.idle_timeout_min * 60.0,
            )
            results.append(result)
            _write_status(status_path, results)
            if result.status != "completed":
                observer_log.write(f"QUEUE STOP after {name} due to status={result.status}\n")
                observer_log.flush()
                return 2
        observer_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] QUEUE COMPLETE\n")
        observer_log.flush()
    _write_status(status_path, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
