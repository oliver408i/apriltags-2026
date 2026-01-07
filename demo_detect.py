import argparse
import sys

import numpy as np

import vision_engine


def load_grayscale(path):
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "No image loader available. Install opencv-python or pillow."
        ) from exc

    return np.array(Image.open(path).convert("L"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="AprilTag demo for vision_engine")
    parser.add_argument("image", help="Path to a grayscale or RGB image")
    parser.add_argument(
        "--family",
        default="tag36h11",
        help="Tag family (e.g. tag36h11, tagStandard41h12)",
    )
    parser.add_argument("--invert", action="store_true", help="Invert grayscale image")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for the image")
    parser.add_argument("--nthreads", type=int, default=None, help="Detector threads")
    parser.add_argument("--quad-decimate", type=float, default=None, help="Quad decimate")
    parser.add_argument("--quad-sigma", type=float, default=None, help="Quad sigma")
    parser.add_argument("--refine-edges", type=int, default=None, help="Refine edges (0/1)")
    parser.add_argument("--decode-sharpening", type=float, default=None, help="Decode sharpening")
    parser.add_argument("--debug-images", action="store_true", help="Write apriltag debug images")
    parser.add_argument("--tag-size", type=float, default=0.162, help="Tag size in meters")
    parser.add_argument("--fx", type=float, default=None, help="Focal length x in pixels")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y in pixels")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x in pixels")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y in pixels")

    args = parser.parse_args(argv)

    image = load_grayscale(args.image)
    height, width = image.shape[:2]

    fx = args.fx if args.fx is not None else float(max(width, height))
    fy = args.fy if args.fy is not None else fx
    cx = args.cx if args.cx is not None else width / 2.0
    cy = args.cy if args.cy is not None else height / 2.0

    vision_engine.set_tag_family(args.family)
    vision_engine.configure_detector(
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=args.refine_edges,
        decode_sharpening=args.decode_sharpening,
        debug=args.debug_images,
    )

    if args.scale and args.scale != 1.0:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required for --scale") from exc
        new_size = (
            max(1, int(image.shape[1] * args.scale)),
            max(1, int(image.shape[0] * args.scale)),
        )
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    if args.invert:
        image = 255 - image
    detections = vision_engine.detect_tags(image, fx, fy, cx, cy, args.tag_size)
    if not detections:
        print("No tags detected.")
        return 1

    for det in detections:
        print(f"tag id={det['id']} hamming={det['hamming']} margin={det['decision_margin']:.2f}")
        print(f"  center={det['center']} corners={det['corners']}")
        print(f"  pose_error={det['pose_error']:.6f}")
        pose = det.get("pose")
        if pose and "rotation" in pose and "translation" in pose:
            rot = np.array(pose["rotation"], dtype=np.float64)
            tvec = pose["translation"]
            roll, pitch, yaw = vision_engine.extract_euler_angles_cython(rot)
            print(f"  tvec={tvec}")
            print(f"  rpy(rad)=({roll:.4f}, {pitch:.4f}, {yaw:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
