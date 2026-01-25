import argparse
import math
import sys

import numpy as np

from tag_map_odometry import compute_camera_pose_from_map, load_tag_map


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    Create a rotation matrix from roll/pitch/yaw (radians).
    """
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotation_matrix_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    if np.linalg.norm(axis) == 0:
        return np.eye(3, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def simulate_detections(tag_map, R_map_camera, t_map_camera, margin, rot_noise, trans_noise):
    """
    Build fake detections for every entry in the tag map based on a camera pose.
    """
    detections = []
    R_camera_map = R_map_camera.T
    for tag_id, entry in tag_map.items():
        R_map_tag = entry["rotation"]
        t_map_tag = entry["translation"]

        det_rot = R_camera_map @ R_map_tag
        det_trans = R_camera_map @ (t_map_tag - t_map_camera)

        if rot_noise > 0.0:
            axis = np.random.normal(size=3)
            angle = abs(np.random.normal(scale=np.deg2rad(rot_noise)))
            det_rot = rotation_matrix_from_axis_angle(axis, angle) @ det_rot

        if trans_noise > 0.0:
            det_trans = det_trans + np.random.normal(scale=trans_noise, size=3)

        detections.append(
            {
                "id": tag_id,
                "decision_margin": margin,
                "pose": {
                    "rotation": det_rot.tolist(),
                    "translation": det_trans.tolist(),
                },
            }
        )
    return detections


def rotation_error_degrees(estimate, ground_truth):
    diff = estimate @ ground_truth.T
    trace = np.clip(np.trace(diff), -1.0, 3.0)
    angle = math.degrees(math.acos((trace - 1.0) / 2.0))
    return angle


def print_header(args):
    print(
        "Simulation:",
        f"frames={args.frames}",
        f"radius={args.radius}",
        f"height={args.height}",
        f"yaw-speed={args.yaw_speed}",
        f"trans-noise={args.translation_noise}",
        f"rot-noise={args.rotation_noise}",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulated multitag odometry tester")
    parser.add_argument("--tag-map", required=True, help="Path to tag map JSON")
    parser.add_argument("--frames", type=int, default=60, help="Number of simulated frames")
    parser.add_argument("--radius", type=float, default=1.0, help="Orbit radius in meters")
    parser.add_argument("--height", type=float, default=0.5, help="Camera height (Z) in meters")
    parser.add_argument("--center-x", type=float, default=0.0, help="Orbit center X")
    parser.add_argument("--center-y", type=float, default=0.0, help="Orbit center Y")
    parser.add_argument("--center-z", type=float, default=0.0, help="Orbit center Z")
    parser.add_argument("--pitch", type=float, default=-15.0, help="Camera pitch angle (deg)")
    parser.add_argument("--roll", type=float, default=0.0, help="Camera roll angle (deg)")
    parser.add_argument("--yaw-speed", type=float, default=10.0, help="Yaw increment (deg) per frame")
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="Starting yaw (deg)")
    parser.add_argument("--decision-margin", type=float, default=10.0, help="Simulated detection margin")
    parser.add_argument("--translation-noise", type=float, default=0.0, help="Translation noise (m std)")
    parser.add_argument("--rotation-noise", type=float, default=0.0, help="Rotation noise (deg std)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = parser.parse_args(argv)

    if args.seed is not None:
        np.random.seed(args.seed)

    tag_map = load_tag_map(args.tag_map)
    print_header(args)

    pitch_rad = math.radians(args.pitch)
    roll_rad = math.radians(args.roll)
    yaw_offset_rad = math.radians(args.yaw_offset)
    yaw_speed_rad = math.radians(args.yaw_speed)

    stats = []
    for frame in range(args.frames):
        angle = yaw_offset_rad + frame * yaw_speed_rad
        cam_x = args.center_x + args.radius * math.cos(angle)
        cam_y = args.center_y + args.radius * math.sin(angle)
        cam_z = args.center_z + args.height
        t_map_camera = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

        yaw_rad = angle
        R_map_camera = rotation_matrix_from_euler(roll_rad, pitch_rad, yaw_rad)

        detections = simulate_detections(
            tag_map,
            R_map_camera,
            t_map_camera,
            args.decision_margin,
            args.rotation_noise,
            args.translation_noise,
        )

        fused = compute_camera_pose_from_map(detections, tag_map)
        if fused is None:
            print(f"frame {frame}: no fused pose")
            continue

        fused_rot = np.asarray(fused["rotation"], dtype=np.float64)
        fused_trans = np.asarray(fused["translation"], dtype=np.float64)
        trans_err = np.linalg.norm(fused_trans - t_map_camera)
        orient_err = rotation_error_degrees(fused_rot, R_map_camera)
        stats.append((trans_err, orient_err))

        print(
            f"frame {frame:02d}: trans_err={trans_err:.3f} m "
            f"orientation_err={orient_err:.2f} deg (tags={len(detections)})"
        )

    if stats:
        trans_errs, orient_errs = zip(*stats)
        print(
            "summary:",
            f"avg_trans_err={np.mean(trans_errs):.3f} m",
            f"avg_orient_err={np.mean(orient_errs):.2f} deg",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
