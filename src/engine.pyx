import atexit

import numpy as np
cimport numpy as cnp
from libc.math cimport atan2, sqrt
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint8_t, uint32_t
from libc.string cimport memcpy

# --- Utility math helpers ---------------------------------------------------

# Migrated from old Numba jit functions, to be used later
def extract_euler_angles_cython(double[:, :] R):
    cdef double sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    cdef bint singular = sy < 1e-6
    cdef double roll, pitch, yaw

    if not singular:
        roll = atan2(R[2, 1], R[2, 2])
        pitch = atan2(-R[2, 0], sy)
        yaw = atan2(R[1, 0], R[0, 0])
    else:
        roll = atan2(-R[1, 2], R[1, 1])
        pitch = atan2(-R[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw

# Ditto
def find_closest_tag_cython(double[:, :] tvecs):
    cdef double min_dist = 1e9
    cdef int best_idx = -1
    cdef int i
    cdef double d

    for i in range(tvecs.shape[0]):
        d = sqrt(tvecs[i, 0]**2 + tvecs[i, 1]**2 + tvecs[i, 2]**2)
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx


# --- C API bindings ---------------------------------------------------------

cdef extern from "apriltag_lib/common/image_u8.h":
    ctypedef struct image_u8_t:
        const int32_t width
        const int32_t height
        const int32_t stride
        uint8_t *buf

    image_u8_t *image_u8_create_stride(unsigned int width, unsigned int height, unsigned int stride)
    image_u8_t *image_u8_create_alignment(unsigned int width, unsigned int height, unsigned int alignment)
    void image_u8_destroy(image_u8_t *im)

cdef extern from "apriltag_lib/common/zarray.h":
    ctypedef struct zarray_t:
        size_t el_sz
        int size
        int alloc
        char *data

cdef extern from "apriltag_lib/common/matd.h":
    ctypedef struct matd_t:
        unsigned int nrows
        unsigned int ncols
        double *data

    void matd_destroy(matd_t *m)

cdef extern from "apriltag_lib/apriltag.h":
    ctypedef struct apriltag_family_t:
        pass

    ctypedef struct apriltag_detector_t:
        pass

    ctypedef struct apriltag_detection_t:
        apriltag_family_t *family
        int id
        int hamming
        float decision_margin
        matd_t *H
        double c[2]
        double p[4][2]

    apriltag_detector_t *apriltag_detector_create()
    void apriltag_detector_add_family(apriltag_detector_t *td, apriltag_family_t *fam)
    void apriltag_detector_destroy(apriltag_detector_t *td)
    zarray_t *apriltag_detector_detect(apriltag_detector_t *td, image_u8_t *im_orig) nogil
    void apriltag_detections_destroy(zarray_t *detections)
    image_u8_t *apriltag_to_image(apriltag_family_t *fam, uint32_t idx)
    pass

cdef extern from "apriltag_lib/tag36h11.h":
    apriltag_family_t *tag36h11_create()
    void tag36h11_destroy(apriltag_family_t *fam)

cdef extern from "apriltag_lib/tag16h5.h":
    apriltag_family_t *tag16h5_create()
    void tag16h5_destroy(apriltag_family_t *fam)

cdef extern from "apriltag_lib/tag25h9.h":
    apriltag_family_t *tag25h9_create()
    void tag25h9_destroy(apriltag_family_t *fam)

cdef extern from "apriltag_lib/tag36h10.h":
    apriltag_family_t *tag36h10_create()
    void tag36h10_destroy(apriltag_family_t *fam)

cdef extern from "apriltag_lib/tagCircle21h7.h":
    apriltag_family_t *tagCircle21h7_create()
    void tagCircle21h7_destroy(apriltag_family_t *fam)

cdef extern from "apriltag_lib/tagCircle49h12.h":
    apriltag_family_t *tagCircle49h12_create()
    void tagCircle49h12_destroy(apriltag_family_t *fam)

cdef extern from "apriltag_lib/tagCustom48h12.h":
    apriltag_family_t *tagCustom48h12_create()
    void tagCustom48h12_destroy(apriltag_family_t *fam)

cdef extern from "apriltag_lib/tagStandard41h12.h":
    apriltag_family_t *tagStandard41h12_create()
    void tagStandard41h12_destroy(apriltag_family_t *fam)

cdef extern from "apriltag_lib/tagStandard52h13.h":
    apriltag_family_t *tagStandard52h13_create()
    void tagStandard52h13_destroy(apriltag_family_t *fam)

cdef extern from "apriltag_lib/apriltag_pose.h":
    ctypedef struct apriltag_detection_info_t:
        apriltag_detection_t *det
        double tagsize
        double fx
        double fy
        double cx
        double cy

    ctypedef struct apriltag_pose_t:
        matd_t *R
        matd_t *t

    double estimate_tag_pose(apriltag_detection_info_t *info, apriltag_pose_t *pose)

cdef extern from *:
    """
    #include "apriltag_lib/apriltag.h"
    #include <stdlib.h>
    #include <string.h>

    static inline void _apriltag_set_nthreads(apriltag_detector_t *td, int nthreads) {
        td->nthreads = nthreads;
    }

    static inline void _apriltag_set_quad_decimate(apriltag_detector_t *td, float quad_decimate) {
        td->quad_decimate = quad_decimate;
    }

    static inline void _apriltag_set_quad_sigma(apriltag_detector_t *td, float quad_sigma) {
        td->quad_sigma = quad_sigma;
    }

    static inline void _apriltag_set_refine_edges(apriltag_detector_t *td, int refine_edges) {
        td->refine_edges = refine_edges;
    }

    static inline void _apriltag_set_decode_sharpening(apriltag_detector_t *td, double decode_sharpening) {
        td->decode_sharpening = decode_sharpening;
    }

    static inline void _apriltag_set_debug(apriltag_detector_t *td, int debug) {
        td->debug = debug;
    }

    static inline image_u8_t *_make_image_u8_header(int width, int height, int stride, uint8_t *buf) {
        image_u8_t tmp = {
            .width = width,
            .height = height,
            .stride = stride,
            .buf = buf
        };
        image_u8_t *im = (image_u8_t *)calloc(1, sizeof(image_u8_t));
        if (!im) {
            return NULL;
        }
        memcpy(im, &tmp, sizeof(image_u8_t));
        return im;
    }

    static inline void _destroy_image_u8_header(image_u8_t *im) {
        free(im);
    }
    """
    void _apriltag_set_nthreads(apriltag_detector_t *td, int nthreads)
    void _apriltag_set_quad_decimate(apriltag_detector_t *td, float quad_decimate)
    void _apriltag_set_quad_sigma(apriltag_detector_t *td, float quad_sigma)
    void _apriltag_set_refine_edges(apriltag_detector_t *td, int refine_edges)
    void _apriltag_set_decode_sharpening(apriltag_detector_t *td, double decode_sharpening)
    void _apriltag_set_debug(apriltag_detector_t *td, int debug)
    image_u8_t *_make_image_u8_header(int width, int height, int stride, uint8_t *buf)
    void _destroy_image_u8_header(image_u8_t *im)

# --- Apriltag detector lifetime ------------------------------------------------

cdef apriltag_detector_t *_apriltag_detector = NULL
cdef apriltag_family_t *_apriltag_family = NULL
cdef str _apriltag_family_name = "tag36h11"
cdef str _apriltag_family_name_active = "tag36h11"
cdef int _cfg_nthreads = -1
cdef float _cfg_quad_decimate = -1.0
cdef float _cfg_quad_sigma = -1.0
cdef int _cfg_refine_edges = -1
cdef double _cfg_decode_sharpening = -1.0
cdef int _cfg_debug = -1


cdef apriltag_family_t *_create_family(str family_name):
    if family_name == "tag16h5":
        return tag16h5_create()
    if family_name == "tag25h9":
        return tag25h9_create()
    if family_name == "tag36h10":
        return tag36h10_create()
    if family_name == "tag36h11":
        return tag36h11_create()
    if family_name == "tagCircle21h7":
        return tagCircle21h7_create()
    if family_name == "tagCircle49h12":
        return tagCircle49h12_create()
    if family_name == "tagCustom48h12":
        return tagCustom48h12_create()
    if family_name == "tagStandard41h12":
        return tagStandard41h12_create()
    if family_name == "tagStandard52h13":
        return tagStandard52h13_create()
    raise ValueError(f"Unsupported tag family: {family_name}")


cdef void _destroy_family(str family_name, apriltag_family_t *family):
    if not family:
        return
    if family_name == "tag16h5":
        tag16h5_destroy(family)
    elif family_name == "tag25h9":
        tag25h9_destroy(family)
    elif family_name == "tag36h10":
        tag36h10_destroy(family)
    elif family_name == "tag36h11":
        tag36h11_destroy(family)
    elif family_name == "tagCircle21h7":
        tagCircle21h7_destroy(family)
    elif family_name == "tagCircle49h12":
        tagCircle49h12_destroy(family)
    elif family_name == "tagCustom48h12":
        tagCustom48h12_destroy(family)
    elif family_name == "tagStandard41h12":
        tagStandard41h12_destroy(family)
    elif family_name == "tagStandard52h13":
        tagStandard52h13_destroy(family)
    else:
        tag36h11_destroy(family)


cdef void _ensure_detector():
    global _apriltag_detector, _apriltag_family

    if _apriltag_detector:
        return

    _apriltag_detector = apriltag_detector_create()
    if not _apriltag_detector:
        raise MemoryError("Unable to allocate Apriltag detector")

    _apriltag_family = _create_family(_apriltag_family_name)
    if not _apriltag_family:
        apriltag_detector_destroy(_apriltag_detector)
        _apriltag_detector = NULL
        raise MemoryError("Unable to create Apriltag family")

    apriltag_detector_add_family(_apriltag_detector, _apriltag_family)
    _apply_detector_config(_apriltag_detector)
    _apriltag_family_name_active = _apriltag_family_name


def set_tag_family(str family_name):
    """
    Change the active AprilTag family. Supported families:
    tag16h5, tag25h9, tag36h10, tag36h11, tagCircle21h7, tagCircle49h12,
    tagCustom48h12, tagStandard41h12, tagStandard52h13.
    """
    global _apriltag_family_name
    if not isinstance(family_name, str):
        raise TypeError("family_name must be a string")

    _apriltag_family_name = family_name

    if _apriltag_detector:
        cleanup_detector()
        _ensure_detector()


def configure_detector(nthreads=None,
                       quad_decimate=None,
                       quad_sigma=None,
                       refine_edges=None,
                       decode_sharpening=None,
                       debug=None):
    """
    Update detector parameters. Use None to keep the current/default setting.
    """
    global _cfg_nthreads, _cfg_quad_decimate, _cfg_quad_sigma
    global _cfg_refine_edges, _cfg_decode_sharpening, _cfg_debug

    if nthreads is not None:
        _cfg_nthreads = int(nthreads)
    if quad_decimate is not None:
        _cfg_quad_decimate = float(quad_decimate)
    if quad_sigma is not None:
        _cfg_quad_sigma = float(quad_sigma)
    if refine_edges is not None:
        _cfg_refine_edges = 1 if refine_edges else 0
    if decode_sharpening is not None:
        _cfg_decode_sharpening = float(decode_sharpening)
    if debug is not None:
        _cfg_debug = 1 if debug else 0

    if _apriltag_detector:
        _apply_detector_config(_apriltag_detector)


def cleanup_detector():
    """Release the shared Apriltag detector and tag family."""
    global _apriltag_detector, _apriltag_family
    detector = _apriltag_detector
    family = _apriltag_family
    family_name = _apriltag_family_name_active
    _apriltag_detector = NULL
    _apriltag_family = NULL

    if detector:
        apriltag_detector_destroy(detector)

    if family:
        _destroy_family(family_name, family)


atexit.register(cleanup_detector)


# --- Helper conversion utilities --------------------------------------------

cdef tuple _matd_to_nested_tuple(matd_t *mat):
    if not mat or not mat.data:
        return ()

    cdef unsigned int rows = mat.nrows
    cdef unsigned int cols = mat.ncols
    cdef unsigned int r, c
    cdef list result = []

    for r in range(rows):
        row_offset = r * cols
        row_list = []
        for c in range(cols):
            row_list.append(mat.data[row_offset + c])
        result.append(tuple(row_list))

    return tuple(result)


cdef tuple _matd_to_flat_tuple(matd_t *mat):
    if not mat or not mat.data:
        return ()

    cdef unsigned int length = mat.nrows * mat.ncols
    cdef unsigned int i
    cdef list result = []
    for i in range(length):
        result.append(mat.data[i])
    return tuple(result)


# --- Apriltag detection binding ----------------------------------------------

def detect_tags(cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] image,
                double fx,
                double fy,
                double cx,
                double cy,
                double tag_size,
                bint copy=True):
    """
    Detect AprilTags in a grayscale image and optionally estimate their pose.

    Parameters
    ----------
    image : ndarray[uint8] (H, W)
        Grayscale image that must be c-contiguous.
    fx, fy : float
        Camera focal lengths in pixels.
    cx, cy : float
        Principal point offsets in pixels.
    tag_size : float
        Physical size of the tag edge (meters). Used for pose estimation.
    copy : bool
        When True, copy the image into an aligned buffer (safer). When False,
        wrap the input buffer directly (faster, but stride/alignment must be valid).

    Returns
    -------
    list of dict
        Each detection contains ``id``, ``hamming``, ``decision_margin``,
        ``center``, ``corners``, ``pose_error``, and ``pose`` when available.
    """

    arr = np.ascontiguousarray(image, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("Detected image must be 2D grayscale")
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] carr = arr

    _ensure_detector()

    cdef int height = <int> carr.shape[0]
    cdef int width = <int> carr.shape[1]
    cdef int stride_bytes = <int> carr.strides[0]
    cdef uint8_t *arr_ptr = &carr[0, 0]
    cdef image_u8_t *frame = NULL
    cdef bint owns_frame = True
    cdef zarray_t *detections
    cdef int y

    if copy:
        frame = image_u8_create_alignment(width, height, 96)
        if not frame:
            raise MemoryError("Unable to allocate Apriltag image buffer")

        for y in range(height):
            memcpy(
                frame.buf + y * frame.stride,
                <void *> (arr_ptr + y * stride_bytes),
                <size_t> width,
            )
    else:
        frame = _make_image_u8_header(width, height, stride_bytes, arr_ptr)
        if not frame:
            raise MemoryError("Unable to allocate Apriltag image header")
        owns_frame = False

    with nogil:
        detections = apriltag_detector_detect(_apriltag_detector, frame)
    if not detections:
        if owns_frame:
            image_u8_destroy(frame)
        else:
            _destroy_image_u8_header(frame)
        return []

    cdef apriltag_detection_t **det_ptr = <apriltag_detection_t **> detections.data
    cdef int count = detections.size
    cdef list results = []
    cdef apriltag_detection_info_t info
    cdef apriltag_pose_t pose
    cdef double pose_error
    cdef dict pose_dict
    info.tagsize = tag_size
    info.fx = fx
    info.fy = fy
    info.cx = cx
    info.cy = cy
    info.det = NULL

    try:
        for idx in range(count):
            det = det_ptr[idx]
            if not det:
                continue

            info.det = det
            pose_error = estimate_tag_pose(&info, &pose)

            center = (det.c[0], det.c[1])
            corners = (
                (det.p[0][0], det.p[0][1]),
                (det.p[1][0], det.p[1][1]),
                (det.p[2][0], det.p[2][1]),
                (det.p[3][0], det.p[3][1]),
            )

            pose_dict = {"error": pose_error}
            if pose.R and pose.t:
                pose_dict["rotation"] = _matd_to_nested_tuple(pose.R)
                pose_dict["translation"] = _matd_to_flat_tuple(pose.t)

            if pose.R:
                matd_destroy(pose.R)

            if pose.t:
                matd_destroy(pose.t)

            results.append({
                "id": det.id,
                "hamming": det.hamming,
                "decision_margin": det.decision_margin,
                "center": center,
                "corners": corners,
                "pose_error": pose_error,
                "pose": pose_dict if len(pose_dict) > 1 else None,
            })
    finally:
        apriltag_detections_destroy(detections)
        if owns_frame:
            image_u8_destroy(frame)
        else:
            _destroy_image_u8_header(frame)

    return results


def generate_tag_image(int tag_id=0):
    """
    Generate a grayscale tag image for the active family.
    """
    _ensure_detector()

    cdef image_u8_t *img = apriltag_to_image(_apriltag_family, <uint32_t> tag_id)
    if not img:
        raise RuntimeError("Failed to generate tag image")

    cdef int height = img.height
    cdef int width = img.width
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] out = np.empty((height, width), dtype=np.uint8)
    cdef uint8_t *out_ptr = &out[0, 0]
    cdef int y
    for y in range(height):
        memcpy(
            out_ptr + y * width,
            img.buf + y * img.stride,
            <size_t> width,
        )

    image_u8_destroy(img)
    return out
cdef void _apply_detector_config(apriltag_detector_t *detector):
    if _cfg_nthreads >= 0:
        _apriltag_set_nthreads(detector, _cfg_nthreads)
    if _cfg_quad_decimate > 0:
        _apriltag_set_quad_decimate(detector, _cfg_quad_decimate)
    if _cfg_quad_sigma >= 0:
        _apriltag_set_quad_sigma(detector, _cfg_quad_sigma)
    if _cfg_refine_edges >= 0:
        _apriltag_set_refine_edges(detector, _cfg_refine_edges)
    if _cfg_decode_sharpening >= 0:
        _apriltag_set_decode_sharpening(detector, _cfg_decode_sharpening)
    if _cfg_debug >= 0:
        _apriltag_set_debug(detector, _cfg_debug)
