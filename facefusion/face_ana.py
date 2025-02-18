# The implementation is adopted from https://github.com/610265158/Peppa_Pig_Face_Engine

import os

import cv2
import numpy as np
#import tensorflow as tf
from easydict import EasyDict as edict
from numpy.linalg import matrix_rank as rank, lstsq, inv, norm

# if tf.__version__ >= '2.0':
#     tf = tf.compat.v1
#     tf.disable_eager_execution()


config = edict()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config.DETECT = edict()
config.DETECT.topk = 10
config.DETECT.thres = 0.8
config.DETECT.input_shape = (512, 512, 3)
config.KEYPOINTS = edict()
config.KEYPOINTS.p_num = 68
config.KEYPOINTS.base_extend_range = [0.2, 0.3]
config.KEYPOINTS.input_shape = (160, 160, 3)
config.TRACE = edict()
config.TRACE.pixel_thres = 1
config.TRACE.smooth_box = 0.3
config.TRACE.smooth_landmark = 0.95
config.TRACE.iou_thres = 0.5
config.DATA = edict()
config.DATA.pixel_means = np.array([123., 116., 103.])  # RGB

# reference facial points, a list of coordinates (x,y)
dx = 1
dy = 1
REFERENCE_FACIAL_POINTS = [
    [30.29459953 + dx, 51.69630051 + dy],  # left eye
    [65.53179932 + dx, 51.50139999 + dy],  # right eye
    [48.02519989 + dx, 71.73660278 + dy],  # nose
    [33.54930115 + dx, 92.3655014 + dy],  # left mouth
    [62.72990036 + dx, 92.20410156 + dy]  # right mouth
]

DEFAULT_CROP_SIZE = (96, 112)

global FACIAL_POINTS


def get_f5p(landmarks, np_img):
    eye_left = find_pupil(landmarks[36:41], np_img)
    eye_right = find_pupil(landmarks[42:47], np_img)
    if eye_left is None or eye_right is None:
        print('cannot find 5 points with find_puil, used mean instead.!')
        eye_left = landmarks[36:41].mean(axis=0)
        eye_right = landmarks[42:47].mean(axis=0)
    nose = landmarks[30]
    mouth_left = landmarks[48]
    mouth_right = landmarks[54]
    f5p = [[eye_left[0], eye_left[1]], [eye_right[0], eye_right[1]],
           [nose[0], nose[1]], [mouth_left[0], mouth_left[1]],
           [mouth_right[0], mouth_right[1]]]
    return f5p


def find_pupil(landmarks, np_img):
    h, w, _ = np_img.shape
    xmax = int(landmarks[:, 0].max())
    xmin = int(landmarks[:, 0].min())
    ymax = int(landmarks[:, 1].max())
    ymin = int(landmarks[:, 1].min())

    if ymin >= ymax or xmin >= xmax or ymin < 0 or xmin < 0 or ymax > h or xmax > w:
        return None
    eye_img_bgr = np_img[ymin:ymax, xmin:xmax, :]
    eye_img = cv2.cvtColor(eye_img_bgr, cv2.COLOR_BGR2GRAY)
    eye_img = cv2.equalizeHist(eye_img)
    n_marks = landmarks - np.array([xmin, ymin]).reshape([1, 2])
    eye_mask = cv2.fillConvexPoly(
        np.zeros_like(eye_img), n_marks.astype(np.int32), 1)
    ret, thresh = cv2.threshold(eye_img, 100, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = (1 - thresh / 255.) * eye_mask
    cnt = 0
    xm = []
    ym = []
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i, j] > 0.5:
                xm.append(j)
                ym.append(i)
                cnt += 1
    if cnt != 0:
        xm.sort()
        ym.sort()
        xm = xm[cnt // 2]
        ym = ym[cnt // 2]
    else:
        xm = thresh.shape[1] / 2
        ym = thresh.shape[0] / 2

    return xm + xmin, ym + ymin


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    h_crop = tmp_crop_size[0]
    w_crop = tmp_crop_size[1]
    if output_size:
        if output_size[0] == h_crop and output_size[1] == w_crop:
            return tmp_5pts

    if inner_padding_factor == 0 and outer_padding == (0, 0):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(
                    tmp_crop_size))

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    factor = inner_padding_factor > 0 or outer_padding[0] > 0
    factor = factor or outer_padding[1] > 0
    if factor and output_size is None:
        output_size = tmp_crop_size * \
                      (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)

    cond1 = outer_padding[0] < output_size[0]
    cond2 = outer_padding[1] < output_size[1]
    if not (cond1 and cond2):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                'and outer_padding[1] < output_size[1])')

    # 1) pad the inner region according inner_padding_factor
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    # 2) resize the padded inner region
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[
        1] * tmp_crop_size[0]:
        raise FaceWarpException(
            'Must have (output_size - outer_padding)'
            '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)

    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rrank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rrank == 3:
        tfm = np.float32([[A[0, 0], A[1, 0], A[2, 0]],
                          [A[0, 1], A[1, 1], A[2, 1]]])
    elif rrank == 2:
        tfm = np.float32([[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0]])

    return tfm


def findNonreflectiveSimilarity(uv, xy):
    options = {'K': 2}

    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    # print('--->x, y:\n', x, y

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    # print('--->X.shape: ', X.shape
    # print('X:\n', X

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # print('--->U.shape: ', U.shape
    # print('U:\n', U

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    # print('--->r:\n', r

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([[sc, -ss, 0], [ss, sc, 0], [tx, ty, 1]])

    # print('--->Tinv:\n', Tinv

    T = inv(Tinv)
    # print('--->T:\n', T

    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv


def tformfwd(trans, uv):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)

    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def findSimilarity(uv, xy, options=None):
    options = {'K': 2}

    #    uv = np.array(uv)
    #    xy = np.array(xy)

    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'trans':
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y, 1] = [u, v, 1] * trans

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        @reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform

    Returns:
    ----------
       @trans: 3x3 np.array
            transform matrix from uv to xy
        trans_inv: 3x3 np.array
            inverse of trans, transform matrix from xy to uv
    """

    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)

    return trans, trans_inv


def cvt_tform_mat_for_cv2(trans):
    """
    Function:
    ----------
        Convert Transform Matrix 'trans' into 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix from uv to xy

    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    cv2_trans = trans[:, 0:2].T

    return cv2_trans


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform

    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)
    cv2_trans_inv = cvt_tform_mat_for_cv2(trans_inv)

    return cv2_trans, cv2_trans_inv


def warp_and_crop_face(src_img,
                       facial_pts,
                       ratio=0.84,
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='similarity'
                                  '',
                       return_trans_inv=False):
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(
                output_size, inner_padding_factor, outer_padding,
                default_square)

    ref_pts = np.float32(reference_pts)

    factor = ratio
    ref_pts = (ref_pts - 112 / 2) * factor + 112 / 2
    ref_pts *= crop_size[0] / 112.

    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts, ref_pts)
        tfm_inv = cv2.getAffineTransform(ref_pts, src_pts)

    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
        tfm_inv = get_affine_transform_matrix(ref_pts, src_pts)
    else:
        tfm, tfm_inv = get_similarity_transform_for_cv2(src_pts, ref_pts)

    face_img = cv2.warpAffine(
        src_img,
        tfm, (crop_size[0], crop_size[1]),
        borderValue=(255, 255, 255))

    if return_trans_inv:
        return face_img, tfm_inv
    else:
        return face_img


class FaceWarpException(Exception):

    def __str__(self):
        return 'In File {}:{}'.format(__file__, super.__str__(self))


class GroupTrack:

    def __init__(self):
        global config
        self.old_frame = None
        self.previous_landmarks_set = None
        self.with_landmark = True
        self.thres = config.TRACE.pixel_thres
        self.alpha = config.TRACE.smooth_landmark
        self.iou_thres = config.TRACE.iou_thres

    def calculate(self, img, current_landmarks_set):
        if self.previous_landmarks_set is None:
            self.previous_landmarks_set = current_landmarks_set
            result = current_landmarks_set
        else:
            previous_lm_num = self.previous_landmarks_set.shape[0]
            if previous_lm_num == 0:
                self.previous_landmarks_set = current_landmarks_set
                result = current_landmarks_set
                return result
            else:
                result = []
                for i in range(current_landmarks_set.shape[0]):
                    not_in_flag = True
                    for j in range(previous_lm_num):
                        if self.iou(current_landmarks_set[i],
                                    self.previous_landmarks_set[j]
                                    ) > self.iou_thres:
                            result.append(
                                self.smooth(current_landmarks_set[i],
                                            self.previous_landmarks_set[j]))
                            not_in_flag = False
                            break
                    if not_in_flag:
                        result.append(current_landmarks_set[i])

        result = np.array(result)
        self.previous_landmarks_set = result

        return result

    def iou(self, p_set0, p_set1):
        rec1 = [
            np.min(p_set0[:, 0]),
            np.min(p_set0[:, 1]),
            np.max(p_set0[:, 0]),
            np.max(p_set0[:, 1])
        ]
        rec2 = [
            np.min(p_set1[:, 0]),
            np.min(p_set1[:, 1]),
            np.max(p_set1[:, 0]),
            np.max(p_set1[:, 1])
        ]

        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        x1 = max(rec1[0], rec2[0])
        y1 = max(rec1[1], rec2[1])
        x2 = min(rec1[2], rec2[2])
        y2 = min(rec1[3], rec2[3])

        # judge if there is an intersect
        intersect = max(0, x2 - x1) * max(0, y2 - y1)

        iou = intersect / (sum_area - intersect)
        return iou

    def smooth(self, now_landmarks, previous_landmarks):
        result = []
        for i in range(now_landmarks.shape[0]):
            x = now_landmarks[i][0] - previous_landmarks[i][0]
            y = now_landmarks[i][1] - previous_landmarks[i][1]
            dis = np.sqrt(np.square(x) + np.square(y))
            if dis < self.thres:
                result.append(previous_landmarks[i])
            else:
                result.append(
                    self.do_moving_average(now_landmarks[i],
                                           previous_landmarks[i]))

        return np.array(result)

    def do_moving_average(self, p_now, p_previous):
        p = self.alpha * p_now + (1 - self.alpha) * p_previous
        return p
