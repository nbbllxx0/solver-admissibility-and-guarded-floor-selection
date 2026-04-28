"""
presets.py
----------
Problem presets for the GPU-FEM pipeline.

Includes all TO3D presets (unchanged) plus GPU-scale presets targeting
problem sizes where GPU acceleration is most beneficial:
  - cantilever_gpu_medium  :  80×40×20 =  64,000 elements
  - cantilever_gpu_large   : 120×60×30 = 216,000 elements
  - cantilever_gpu_xlarge  : 160×80×40 = 512,000 elements (needs 8+GB VRAM)
  - mbb_gpu_large          : 200×50×25 = 250,000 elements
  - bridge_gpu_large       : 150×50×25 = 187,500 elements

GPU-scale presets use the same physical setup as their small counterparts
to allow direct compliance comparison.
"""

from __future__ import annotations

from .problem_spec import (
    ProblemSpec,
    EdgeSupport,
    PointSupport,
    PointLoad,
    DistributedLoad,
)


# ─────────────────────────────────────────────────────────────────────────────
# Standard 3D Presets (copied from TO3D presets.py — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

PRESETS_3D: dict[str, ProblemSpec] = {

    "cantilever_3d": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=24, nely=12, nelz=6,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    "bridge_3d": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=30, nely=10, nelz=5,
        volfrac=0.3,
        supports=[
            PointSupport(x=0.0, y=0.0, z=0.0,  constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.25, constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.5,  constraint="fixed"),
            PointSupport(x=3.0, y=0.0, z=0.0,  constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.25, constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.5,  constraint="roller_x"),
        ],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    "mbb_3d": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=30, nely=10, nelz=5,
        volfrac=0.5,
        supports=[
            EdgeSupport(edge="left", constraint="pin_x"),
            PointSupport(x=3.0, y=0.0, z=0.25, constraint="pin_y"),
        ],
        loads=[PointLoad(x=0.0, y=1.0, z=0.25, fy=-1.0)],
    ),

    "bracket_3d": ProblemSpec(
        Lx=1.0, Ly=2.0, Lz=0.5,
        nelx=12, nely=24, nelz=6,
        volfrac=0.3,
        supports=[EdgeSupport(edge="top", constraint="fixed")],
        loads=[PointLoad(x=1.0, y=0.0, z=0.25, fy=-0.7, fx=0.3)],
    ),

    "torsion_3d": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=1.0,
        nelx=30, nely=10, nelz=10,
        volfrac=0.25,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[
            PointLoad(x=3.0, y=1.0, z=0.5, fy=-0.5, fz=0.5),
            PointLoad(x=3.0, y=0.0, z=0.5, fy=0.5,  fz=-0.5),
        ],
    ),

    "column_3d": ProblemSpec(
        Lx=1.0, Ly=4.0, Lz=1.0,
        nelx=10, nely=40, nelz=10,
        volfrac=0.20,
        supports=[EdgeSupport(edge="bottom", constraint="fixed")],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    # JMD additions
    "cantilever_3d_large": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=40, nely=20, nelz=10,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    "cantilever_3d_xlarge": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=60, nely=30, nelz=15,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    "cantilever_3d_vf02": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=24, nely=12, nelz=6,
        volfrac=0.20,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    "cantilever_3d_vf04": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=24, nely=12, nelz=6,
        volfrac=0.40,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# GPU-Scale Presets (NEW — sized for GPU acceleration studies)
# ─────────────────────────────────────────────────────────────────────────────

PRESETS_GPU: dict[str, ProblemSpec] = {

    # 80×40×20 = 64,000 elements — "medium GPU" (good starting point)
    "cantilever_gpu_medium": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=80, nely=40, nelz=20,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    # 120×60×30 = 216,000 elements — "large GPU" (primary benchmark target)
    "cantilever_gpu_large": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=120, nely=60, nelz=30,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    # 160×80×40 = 512,000 elements — stress-test, needs 8+ GB VRAM
    "cantilever_gpu_xlarge": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=160, nely=80, nelz=40,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    # 200×100×50 = 1,000,000 elements — first 1M-element 3D SIMP (matrix-free only)
    "cantilever_gpu_xxlarge": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=200, nely=100, nelz=50,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    # 252×126×63 = 2,000,376 elements — ~2M ablation (matrix-free only, ~5 GB)
    "cantilever_gpu_2M": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=252, nely=126, nelz=63,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    # 342×171×86 = 5,032,692 elements — ~5M ablation (matrix-free only, ~11 GB)
    "cantilever_gpu_5M": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=342, nely=171, nelz=86,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    # 400×200×100 = 8,000,000 elements — 8M ablation (matrix-free only, ~17 GB)
    "cantilever_gpu_8M": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=400, nely=200, nelz=100,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    # 430×215×108 = 9,989,400 elements — ~10M ablation (matrix-free only, ~21 GB, tight)
    "cantilever_gpu_10M": ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=430, nely=215, nelz=108,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    ),

    # 210×70×35 = 514,500 elements — MBB at ~500k (matrix-free only)
    "mbb_gpu_xlarge": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=210, nely=70, nelz=35,
        volfrac=0.5,
        supports=[
            EdgeSupport(edge="left", constraint="pin_x"),
            PointSupport(x=3.0, y=0.0, z=0.25, constraint="pin_y"),
        ],
        loads=[PointLoad(x=0.0, y=1.0, z=0.25, fy=-1.0)],
    ),

    # 264×88×44 = 1,021,824 elements — MBB at ~1M (matrix-free only)
    "mbb_gpu_xxlarge": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=264, nely=88, nelz=44,
        volfrac=0.5,
        supports=[
            EdgeSupport(edge="left", constraint="pin_x"),
            PointSupport(x=3.0, y=0.0, z=0.25, constraint="pin_y"),
        ],
        loads=[PointLoad(x=0.0, y=1.0, z=0.25, fy=-1.0)],
    ),

    # 200×50×25 = 250,000 elements — MBB variant at GPU scale
    "mbb_gpu_large": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=200, nely=50, nelz=25,
        volfrac=0.5,
        supports=[
            EdgeSupport(edge="left", constraint="pin_x"),
            PointSupport(x=3.0, y=0.0, z=0.25, constraint="pin_y"),
        ],
        loads=[PointLoad(x=0.0, y=1.0, z=0.25, fy=-1.0)],
    ),

    # 150×50×25 = 187,500 elements — bridge at GPU scale
    "bridge_gpu_large": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=150, nely=50, nelz=25,
        volfrac=0.3,
        supports=[
            PointSupport(x=0.0, y=0.0, z=0.0,   constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.125,  constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.25,   constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.375,  constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.5,    constraint="fixed"),
            PointSupport(x=3.0, y=0.0, z=0.0,    constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.125,  constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.25,   constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.375,  constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.5,    constraint="roller_x"),
        ],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    # Bridge family with a distributed load and fixed/roller supports
    "bridge_gpu_medium": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=90, nely=30, nelz=15,
        volfrac=0.3,
        supports=[
            PointSupport(x=0.0, y=0.0, z=0.0,   constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.25,  constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.5,   constraint="fixed"),
            PointSupport(x=3.0, y=0.0, z=0.0,   constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.25,  constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.5,   constraint="roller_x"),
        ],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    "bridge_gpu_xlarge": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=180, nely=60, nelz=30,
        volfrac=0.3,
        supports=[
            PointSupport(x=0.0, y=0.0, z=0.0,   constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.25,  constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.5,   constraint="fixed"),
            PointSupport(x=3.0, y=0.0, z=0.0,   constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.25,  constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.5,   constraint="roller_x"),
        ],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    # 210×70×35 = 514,500 elements — Bridge at ~500k (matrix-free only)
    "bridge_gpu_500k": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=210, nely=70, nelz=35,
        volfrac=0.3,
        supports=[
            PointSupport(x=0.0, y=0.0, z=0.0,    constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.25,   constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.5,    constraint="fixed"),
            PointSupport(x=3.0, y=0.0, z=0.0,    constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.25,   constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.5,    constraint="roller_x"),
        ],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    # 264×88×44 = 1,021,824 elements — Bridge at ~1M (matrix-free only)
    "bridge_gpu_1M": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=264, nely=88, nelz=44,
        volfrac=0.3,
        supports=[
            PointSupport(x=0.0, y=0.0, z=0.0,    constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.25,   constraint="fixed"),
            PointSupport(x=0.0, y=0.0, z=0.5,    constraint="fixed"),
            PointSupport(x=3.0, y=0.0, z=0.0,    constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.25,   constraint="roller_x"),
            PointSupport(x=3.0, y=0.0, z=0.5,    constraint="roller_x"),
        ],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    # MBB family with a concentrated downward load and mixed supports
    "mbb_gpu_medium": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=90, nely=30, nelz=15,
        volfrac=0.5,
        supports=[
            EdgeSupport(edge="left", constraint="pin_x"),
            PointSupport(x=3.0, y=0.0, z=0.25, constraint="pin_y"),
        ],
        loads=[PointLoad(x=0.0, y=1.0, z=0.25, fy=-1.0)],
    ),

    "mbb_gpu_large": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=0.5,
        nelx=150, nely=50, nelz=25,
        volfrac=0.5,
        supports=[
            EdgeSupport(edge="left", constraint="pin_x"),
            PointSupport(x=3.0, y=0.0, z=0.25, constraint="pin_y"),
        ],
        loads=[PointLoad(x=0.0, y=1.0, z=0.25, fy=-1.0)],
    ),

    # Torsion family with opposing tangential loads
    "torsion_gpu_medium": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=1.0,
        nelx=60, nely=20, nelz=20,
        volfrac=0.25,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[
            PointLoad(x=3.0, y=1.0, z=0.5, fy=-0.5, fz=0.5),
            PointLoad(x=3.0, y=0.0, z=0.5, fy=0.5, fz=-0.5),
        ],
    ),

    # Bracket family with mixed axial/lateral loading
    "bracket_gpu_medium": ProblemSpec(
        Lx=1.0, Ly=2.0, Lz=0.5,
        nelx=30, nely=60, nelz=15,
        volfrac=0.3,
        supports=[EdgeSupport(edge="top", constraint="fixed")],
        loads=[PointLoad(x=1.0, y=0.0, z=0.25, fy=-0.7, fx=0.3)],
    ),

    # Column family with distributed compression
    "column_gpu_medium": ProblemSpec(
        Lx=1.0, Ly=4.0, Lz=1.0,
        nelx=20, nely=80, nelz=20,
        volfrac=0.20,
        supports=[EdgeSupport(edge="bottom", constraint="fixed")],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    # ── Large-scale presets for Paper 2 problem variety ──────────────────────

    # Torsion: 165×55×55 = 498,375 elems, ~500k
    "torsion_gpu_500k": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=1.0,
        nelx=165, nely=55, nelz=55,
        volfrac=0.25,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[
            PointLoad(x=3.0, y=1.0, z=0.5, fy=-0.5, fz=0.5),
            PointLoad(x=3.0, y=0.0, z=0.5, fy=0.5,  fz=-0.5),
        ],
    ),

    # Torsion: 210×70×70 = 1,029,000 elems, ~1M
    "torsion_gpu_1M": ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=1.0,
        nelx=210, nely=70, nelz=70,
        volfrac=0.25,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[
            PointLoad(x=3.0, y=1.0, z=0.5, fy=-0.5, fz=0.5),
            PointLoad(x=3.0, y=0.0, z=0.5, fy=0.5,  fz=-0.5),
        ],
    ),

    # Column: 50×200×50 = 500,000 elems, ~500k
    "column_gpu_500k": ProblemSpec(
        Lx=1.0, Ly=4.0, Lz=1.0,
        nelx=50, nely=200, nelz=50,
        volfrac=0.20,
        supports=[EdgeSupport(edge="bottom", constraint="fixed")],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    # Column: 63×252×63 = 999,972 elems, ~1M
    "column_gpu_1M": ProblemSpec(
        Lx=1.0, Ly=4.0, Lz=1.0,
        nelx=63, nely=252, nelz=63,
        volfrac=0.20,
        supports=[EdgeSupport(edge="bottom", constraint="fixed")],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),

    # Bracket: 80×160×40 = 512,000 elems, ~500k
    "bracket_gpu_500k": ProblemSpec(
        Lx=1.0, Ly=2.0, Lz=0.5,
        nelx=80, nely=160, nelz=40,
        volfrac=0.30,
        supports=[EdgeSupport(edge="top", constraint="fixed")],
        loads=[PointLoad(x=1.0, y=0.0, z=0.25, fy=-0.7, fx=0.3)],
    ),

    # Bracket: 100×200×50 = 1,000,000 elems, ~1M
    "bracket_gpu_1M": ProblemSpec(
        Lx=1.0, Ly=2.0, Lz=0.5,
        nelx=100, nely=200, nelz=50,
        volfrac=0.30,
        supports=[EdgeSupport(edge="top", constraint="fixed")],
        loads=[PointLoad(x=1.0, y=0.0, z=0.25, fy=-0.7, fx=0.3)],
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Standard 2D Presets (from TO3D presets.py — for fast testing)
# ─────────────────────────────────────────────────────────────────────────────

PRESETS_2D: dict[str, ProblemSpec] = {

    "cantilever": ProblemSpec(
        Lx=2.0, Ly=1.0,
        nelx=60, nely=30,
        volfrac=0.5,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, fy=-1.0)],
    ),

    "mbb": ProblemSpec(
        Lx=3.0, Ly=1.0,
        nelx=90, nely=30,
        volfrac=0.5,
        supports=[
            EdgeSupport(edge="left", constraint="pin_x"),
            PointSupport(x=3.0, y=0.0, constraint="pin_y"),
        ],
        loads=[PointLoad(x=0.0, y=1.0, fy=-1.0)],
    ),

    "bridge": ProblemSpec(
        Lx=4.0, Ly=1.0,
        nelx=120, nely=30,
        volfrac=0.3,
        supports=[
            PointSupport(x=0.0, y=0.0, constraint="fixed"),
            PointSupport(x=4.0, y=0.0, constraint="roller_x"),
        ],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_PRESETS: dict[str, ProblemSpec] = {**PRESETS_3D, **PRESETS_GPU, **PRESETS_2D}


def get_preset(name: str) -> ProblemSpec:
    if name not in ALL_PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {sorted(ALL_PRESETS.keys())}"
        )
    return ALL_PRESETS[name]


def list_presets() -> list[str]:
    return sorted(ALL_PRESETS.keys())


def list_gpu_presets() -> list[str]:
    return sorted(PRESETS_GPU.keys())


def n_elements(name: str) -> int:
    s = get_preset(name)
    return s.nelx * s.nely * (s.nelz if s.nelz > 0 else 1)
