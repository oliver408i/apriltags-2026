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

## Benchmarks
The following benchmarks were conducted using a synthetic 720p (1280x720) test suite to isolate computational latency from camera/USB bus overhead. 

| Processor | Architecture | Threads | Latency (Det) | Max FPS |
| :--- | :--- | :---: | :---: | :---: |
| **Ryzen 7 7700X** | x86 (Zen 4) | 15 | **5.1 ms** | **191.3** |
| Ryzen 7 7700X | x86 (Zen 4) | 4 | 7.6 ms | 128.3 |
| **i5-1245U** | x86 (Alder Lake) | 12 | **9.4 ms** | **100.8** |
| i5-1245U | x86 (Alder Lake) | 4 | 11.8 ms | 81.8 |
| **Ryzen 3 4300U** | x86 (Zen 2) | 4 | **17.0 ms** | **56.1** |
| Ryzen 3 4300U | x86 (Zen 2) | 1 | 41.5 ms | 23.7 |
| **Cortex-A78AE** | ARM (v8.2) | 4 | **20.2 ms** | **47.4** |
| Cortex-A78AE | ARM (v8.2) | 8 | 21.1 ms | 44.9 |
| **Cortex-A76** | **ARM (Pi 5)** | **4** | **21.8 ms** | **43.2** |
| Cortex-A76 | ARM (Pi 5) | 1 | 60.2 ms | 16.3 |
| **i3-10110U** | x86 (Comet Lake) | 4 | **23.3 ms** | **41.2** |
| i3-10110U | x86 (Comet Lake) | 1 | 43.5 ms | 22.4 |
| **Cortex-A72** | **ARM (Pi 4B)** | **4** | **59.1 ms** | **16.2** |
| Cortex-A72 | ARM (Pi 4B) | 1 | 146.0 ms | 6.7 |

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

## Multitag Odometry

`demo_live.py` can optionally load a static tag map via `--tag-map path/to/your_map.json`. The map should describe each tag's pose in a shared coordinate frame and can be either a dictionary or a list of entries. Each entry needs an `id`, a `translation` as `[x, y, z]` in meters, and a `rotation` as a 3×3 matrix (a flat list of nine numbers is also accepted). For example:

```json
{
  "0": {
    "translation": [0.0, 0.0, 0.0],
    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  },
  "1": {
    "translation": [0.5, 0.0, 0.0],
    "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
  }
}
```

When a tag map is provided, detections that match the defined tags are fused into a single camera pose estimate. The pose is rendered on the display overlay and appended to the `--fps` telemetry line when present.

A sample map is available at `tag_map_sample.json` if you need a quick configuration to test the odometry flow.

## Simulator

`demo_simulator.py` lets you validate the multitag odometry flow without a live camera. Supply the same JSON tag map (`--tag-map`) and the simulator will move a virtual camera around the map while emitting fake detections, computing the fused pose, and printing translation/orientation errors. Example:

```
python demo_simulator.py --tag-map path/to/map.json --frames 120 --radius 1.2 --rotation-noise 0.5
```

Use `--translation-noise` or `--rotation-noise` to stress-test the fusion with measurement noise and `--seed` for repeatable runs.

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
