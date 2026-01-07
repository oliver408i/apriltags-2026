import argparse
import time

import vision_engine


def main(argv=None):
    parser = argparse.ArgumentParser(description="AprilTag self-test for vision_engine")
    parser.add_argument("--family", default="tag36h11", help="Tag family name")
    parser.add_argument("--tag-id", type=int, default=0, help="Tag id to generate")
    parser.add_argument("--tag-size", type=float, default=0.162, help="Tag size in meters")
    parser.add_argument("--scale", type=int, default=8, help="Scale factor for detection")
    parser.add_argument("--invert", action="store_true", help="Invert grayscale image")
    parser.add_argument("--runs", type=int, default=10, help="Timing iterations per mode")
    args = parser.parse_args(argv)

    vision_engine.set_tag_family(args.family)
    vision_engine.configure_detector(
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )
    image = vision_engine.generate_tag_image(args.tag_id)
    if args.scale and args.scale > 1:
        image = image.repeat(args.scale, axis=0).repeat(args.scale, axis=1)
    if args.invert:
        image = 255 - image

    height, width = image.shape[:2]
    fx = float(max(width, height))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    def run_mode(copy_flag):
        t0 = time.perf_counter()
        detections = None
        for _ in range(args.runs):
            detections = vision_engine.detect_tags(
                image, fx, fy, cx, cy, args.tag_size, copy=copy_flag
            )
        t1 = time.perf_counter()
        return detections or [], (t1 - t0) / max(1, args.runs)

    detections_copy, avg_copy = run_mode(True)
    detections_nocopy, avg_nocopy = run_mode(False)

    print(
        f"Generated {args.family} id={args.tag_id} -> "
        f"copy detections: {len(detections_copy)} "
        f"({avg_copy*1000:.2f} ms avg)"
    )
    print(
        f"Generated {args.family} id={args.tag_id} -> "
        f"no-copy detections: {len(detections_nocopy)} "
        f"({avg_nocopy*1000:.2f} ms avg)"
    )
    if detections_copy:
        print("Example detection (copy):")
        print(detections_copy[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
