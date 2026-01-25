import argparse
import os
import sys
import time

import cv2
import numpy as np

import vision_engine


def _load_intrinsics(args, width, height):
    fx = args.fx
    fy = args.fy
    cx = args.cx
    cy = args.cy

    if args.camera_matrix and os.path.exists(args.camera_matrix):
        cam = np.load(args.camera_matrix)
        if cam.shape[0] >= 3 and cam.shape[1] >= 3:
            fx = float(cam[0, 0])
            fy = float(cam[1, 1])
            cx = float(cam[0, 2])
            cy = float(cam[1, 2])
        else:
            raise ValueError("camera_matrix.npy must be at least 3x3")

    if fx is None:
        fx = float(max(width, height))
    if fy is None:
        fy = fx
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    return fx, fy, cx, cy


def _open_capture(args):
    if args.synthetic:
        return None
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        if args.backend == "v4l2":
            cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(args.camera)

        if args.fourcc:
            fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.camera_fps:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    if not cap.isOpened():
        return None

    return cap


def _print_capture_info(cap):
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(
        [
            chr((actual_fourcc >> 0) & 0xFF),
            chr((actual_fourcc >> 8) & 0xFF),
            chr((actual_fourcc >> 16) & 0xFF),
            chr((actual_fourcc >> 24) & 0xFF),
        ]
    )
    print(
        "Capture settings:",
        f"{actual_width}x{actual_height}",
        f"{actual_fps:.2f} FPS",
        f"FOURCC={fourcc_str}",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Headless AprilTag benchmark")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--video", default=None, help="Optional video file path")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic frames instead of a camera/video",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "v4l2"),
        default="auto",
        help="OpenCV capture backend (auto or v4l2)",
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--camera-fps", type=int, default=30, help="Capture FPS request")
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="FourCC for capture (e.g. MJPG, YUYV, H264)",
    )
    parser.add_argument("--tag-size", type=float, default=0.120, help="Tag size in meters")
    parser.add_argument(
        "--family",
        default="tag36h11",
        help="Tag family (e.g. tag36h11, tagStandard41h12)",
    )
    parser.add_argument("--nthreads", type=int, default=None, help="Detector threads")
    parser.add_argument("--quad-decimate", type=float, default=None, help="Quad decimate")
    parser.add_argument("--quad-sigma", type=float, default=None, help="Quad sigma")
    parser.add_argument("--refine-edges", type=int, default=None, help="Refine edges (0/1)")
    parser.add_argument("--decode-sharpening", type=float, default=None, help="Decode sharpening")
    parser.add_argument("--debug-images", action="store_true", help="Write apriltag debug images")
    parser.add_argument("--invert", action="store_true", help="Invert grayscale image")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for detection")
    parser.add_argument("--camera-matrix", default="camera_matrix.npy", help="Path to camera matrix")
    parser.add_argument("--fx", type=float, default=None, help="Focal length x in pixels")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y in pixels")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x in pixels")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y in pixels")
    parser.add_argument("--duration", type=float, default=10.0, help="Benchmark duration (s)")
    parser.add_argument("--warmup", type=float, default=1.0, help="Warmup duration (s)")
    parser.add_argument(
        "--report-interval",
        type=float,
        default=1.0,
        help="Seconds between reports",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Avoid copying the input frame (requires contiguous data)",
    )

    args = parser.parse_args(argv)

    cap = _open_capture(args)
    if not args.synthetic:
        if cap is None:
            print("Failed to open capture.", file=sys.stderr)
            return 1
        _print_capture_info(cap)

    vision_engine.set_tag_family(args.family)
    vision_engine.configure_detector(
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=args.refine_edges,
        decode_sharpening=args.decode_sharpening,
        debug=args.debug_images,
    )

    total_frames = 0
    total_det = 0
    total_cap_ms = 0.0
    total_pre_ms = 0.0
    total_det_ms = 0.0

    interval_frames = 0
    interval_det = 0
    interval_cap_ms = 0.0
    interval_pre_ms = 0.0
    interval_det_ms = 0.0
    interval_start = time.perf_counter()

    start = time.perf_counter()
    warmup_end = start + max(0.0, args.warmup)
    stop_at = start + max(0.0, args.duration)

    synth_gray = None
    synth_color = None
    if args.synthetic:
        synth_gray = np.random.randint(
            0, 256, (args.height, args.width), dtype=np.uint8
        )
        synth_color = cv2.cvtColor(synth_gray, cv2.COLOR_GRAY2BGR)

    while True:
        now = time.perf_counter()
        if now >= stop_at:
            break

        cap_start = time.perf_counter()
        if args.synthetic:
            frame = synth_color
            ok = True
        else:
            ok, frame = cap.read()
        cap_ms = (time.perf_counter() - cap_start) * 1000.0
        if not ok:
            print("Failed to read frame.", file=sys.stderr)
            break

        pre_start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.scale and args.scale != 1.0:
            new_size = (
                max(1, int(gray.shape[1] * args.scale)),
                max(1, int(gray.shape[0] * args.scale)),
            )
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_LINEAR)
        if args.invert:
            gray = 255 - gray
        height, width = gray.shape[:2]
        pre_ms = (time.perf_counter() - pre_start) * 1000.0

        fx, fy, cx, cy = _load_intrinsics(args, width, height)
        det_start = time.perf_counter()
        detections = vision_engine.detect_tags(
            gray, fx, fy, cx, cy, args.tag_size, copy=not args.no_copy
        )
        det_ms = (time.perf_counter() - det_start) * 1000.0

        if now >= warmup_end:
            total_frames += 1
            total_det += len(detections)
            total_cap_ms += cap_ms
            total_pre_ms += pre_ms
            total_det_ms += det_ms

            interval_frames += 1
            interval_det += len(detections)
            interval_cap_ms += cap_ms
            interval_pre_ms += pre_ms
            interval_det_ms += det_ms

        if now - interval_start >= args.report_interval:
            elapsed = now - interval_start
            if interval_frames > 0 and elapsed > 0:
                fps = interval_frames / elapsed
                print(
                    f"FPS: {fps:.1f}  det/frame: {interval_det / interval_frames:.2f}  "
                    f"cap={interval_cap_ms / interval_frames:.1f}ms  "
                    f"pre={interval_pre_ms / interval_frames:.1f}ms  "
                    f"det={interval_det_ms / interval_frames:.1f}ms"
                )
            interval_frames = 0
            interval_det = 0
            interval_cap_ms = 0.0
            interval_pre_ms = 0.0
            interval_det_ms = 0.0
            interval_start = now

    if cap is not None:
        cap.release()

    total_time = time.perf_counter() - max(warmup_end, start)
    if total_frames > 0 and total_time > 0:
        print(
            "Overall:",
            f"FPS={total_frames / total_time:.1f}",
            f"det/frame={total_det / total_frames:.2f}",
            f"cap={total_cap_ms / total_frames:.1f}ms",
            f"pre={total_pre_ms / total_frames:.1f}ms",
            f"det={total_det_ms / total_frames:.1f}ms",
        )
    else:
        print("No frames processed after warmup.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
