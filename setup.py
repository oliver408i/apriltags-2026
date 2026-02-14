# setup.py
from pathlib import Path
import os

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ROOT = Path(__file__).resolve().parent
APRILTAG_ROOT = ROOT / "src" / "apriltag_lib"
APRILTAG_COMMON = APRILTAG_ROOT / "common"
APRILTAG_LIB = ROOT / "build" / "apriltag" / "libapriltag.a"
USE_PREBUILT = os.environ.get("USE_PREBUILT_APRILTAG") == "1"

def _collect_apriltag_sources():
    core_files = ["apriltag.c", "apriltag_pose.c", "apriltag_quad_thresh.c"]
    tags = sorted(p for p in APRILTAG_ROOT.glob("tag*.c") if p.is_file())
    common_files = sorted(p for p in APRILTAG_COMMON.glob("*.c") if p.is_file())
    paths = [APRILTAG_ROOT / fn for fn in core_files] + list(tags) + list(common_files)
    # Setuptools disallows absolute paths in sources; keep them relative to setup.py.
    return [str(p.relative_to(ROOT)) for p in paths]

apriltag_sources = _collect_apriltag_sources()

if USE_PREBUILT and APRILTAG_LIB.exists():
    apriltag_build_sources = ["src/engine.pyx"]
    apriltag_extra_objects = [str(APRILTAG_LIB.relative_to(ROOT))]
else:
    apriltag_build_sources = ["src/engine.pyx"] + apriltag_sources
    apriltag_extra_objects = []

extensions = [
    Extension(
        "vision_engine",
        sources=apriltag_build_sources,
        include_dirs=[
            np.get_include(),
            "src",
            "src/apriltag_lib",
            "src/apriltag_lib/common",
        ],
        libraries=["pthread", "m"],
        extra_objects=apriltag_extra_objects,
        extra_compile_args=[
            "-O3",
            "-std=c11",
            "-D_GNU_SOURCE",
            "-D_DEFAULT_SOURCE",
        ],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        force=True,
    ),
)
