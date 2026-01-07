# AprilTag Cython Notes
AprilTag detector from 2025 but its rewritten in Cython instead of using Numba. TL;DR: It's fast

## Structure
- `src/engine.pyx`: Cython bindings + detection helpers.
- `src/apriltag_lib/`: AprilTag C library source (vendored).
- `setup.py`: builds the `vision_engine` extension.
- `Makefile`: top-level build (Apriltag + Cython).
- `demo_detect.py`: run detection on a single image.
- `demo_live.py`: run detection on a live camera feed.
- `demo_selftest.py`: generate a tag image and detect it (sanity check).

## Downloading
**IMPORTANT:** This repo uses the `apriltag_lib` submodule! Remember to do `git submodule update --init --recursive`

## Build
```
make
```

This runs:
- CMake in `build/apriltag` to build `libapriltag.a`
- `python setup.py build_ext --inplace` to build `vision_engine`

If you want to rebuild only the extension:
```
make cython
```

## Demos
Detect from an image:
```
python demo_detect.py path/to/image.png --family tag36h11 --tag-size 0.162
```

Live camera:
```
python demo_live.py --display --family tag36h11 --tag-size 0.162
```

Self-test:
```
python demo_selftest.py --family tag36h11 --tag-id 0
```

## Camera intrinsics
If you have intrinsics:
- `camera_matrix.npy` (3x3)
- `dist_coeffs.npy` (vector)

`demo_live.py` loads these automatically if present.

## Helpful knobs
Both demos accept detector tuning flags:
- `--quad-decimate 1.0`
- `--quad-sigma 0.0`
- `--refine-edges 1`
- `--decode-sharpening 0.25`

If detections are missing, try `--scale` and `--invert` too.

## Python API
`vision_engine` exports these functions:

- `extract_euler_angles_cython(R)`  
  Convert a 3x3 rotation matrix into roll/pitch/yaw (radians).

- `find_closest_tag_cython(tvecs)`  
  Pick the closest tag index from a list of translation vectors.

- `detect_tags(image, fx, fy, cx, cy, tag_size, copy=True)`  
  Run AprilTag detection on a grayscale `uint8` image. Returns a list of dicts:
  - `id`, `hamming`, `decision_margin`
  - `center`, `corners`
  - `pose_error`
  - `pose` (dict with `rotation` 3x3 and `translation` 3x1) or `None`

- `set_tag_family(name)`  
  Supported: `tag16h5`, `tag25h9`, `tag36h10`, `tag36h11`,
  `tagCircle21h7`, `tagCircle49h12`, `tagCustom48h12`,
  `tagStandard41h12`, `tagStandard52h13`

- `configure_detector(...)`  
  Keyword args: `nthreads`, `quad_decimate`, `quad_sigma`,
  `refine_edges`, `decode_sharpening`, `debug`

- `generate_tag_image(tag_id=0)`  
  Generate a tag image for the active family as a grayscale NumPy array.
