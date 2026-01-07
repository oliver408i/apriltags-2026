# setup.py
from pathlib import Path

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ROOT = Path(__file__).resolve().parent
APRILTAG_ROOT = ROOT / "src" / "apriltag_lib"
APRILTAG_COMMON = APRILTAG_ROOT / "common"
APRILTAG_LIB = ROOT / "build" / "apriltag" / "libapriltag.a"

def _collect_apriltag_sources():
    core_files = ["apriltag.c", "apriltag_pose.c", "apriltag_quad_thresh.c"]
    tags = sorted(p for p in APRILTAG_ROOT.glob("tag*.c") if p.is_file())
    common_files = sorted(p for p in APRILTAG_COMMON.glob("*.c") if p.is_file())
    return [str(APRILTAG_ROOT / fn) for fn in core_files] + [str(p) for p in tags + common_files]

apriltag_sources = _collect_apriltag_sources()

if APRILTAG_LIB.exists():
    apriltag_build_sources = ["src/engine.pyx"]
    apriltag_extra_objects = [str(APRILTAG_LIB)]
else:
    apriltag_build_sources = ["src/engine.pyx"] + apriltag_sources
    apriltag_extra_objects = []

extensions = [
    Extension(
        "vision_engine",
        sources=apriltag_build_sources,
        include_dirs=[np.get_include(), str(ROOT / "src")],
        libraries=["pthread", "m"],
        extra_objects=apriltag_extra_objects,
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-std=c11",
            "-D_GNU_SOURCE",
            "-D_DEFAULT_SOURCE",
        ],
    )
]

setup(
    name="VisionEngine",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
