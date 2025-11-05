import math
from numpy import random
import cv2
import numpy as np
import torch


def rnd(x):
    return max(-2 * x, min(2 * x, np.random.randn(1)[0] * x))


def box_transform(bbox, sf, imgwidth, imght, train):
    """Random scaling."""
    width = bbox[2] - bbox[0]
    ht = bbox[3] - bbox[1]
    if train:
        scaleRate = 0.25 * np.clip(np.random.randn() * sf, - sf, sf)

        bbox[0] = max(0, bbox[0] - width * scaleRate / 2)
        bbox[1] = max(0, bbox[1] - ht * scaleRate / 2)
        bbox[2] = min(imgwidth, bbox[2] + width * scaleRate / 2)
        bbox[3] = min(imght, bbox[3] + ht * scaleRate / 2)
    else:
        scaleRate = 0.25

        bbox[0] = max(0, bbox[0] - width * scaleRate / 2)
        bbox[1] = max(0, bbox[1] - ht * scaleRate / 2)
        bbox[2] = min(imgwidth, max(bbox[2] + width * scaleRate / 2, bbox[0] + 5))
        bbox[3] = min(imght, max(bbox[3] + ht * scaleRate / 2, bbox[1] + 5))

    return bbox


def im_to_torch(img):
    """Transform ndarray image to torch tensor.

    Parameters
    ----------
    img: numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.

    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_im(img):
    """Transform torch tensor to ndarray image.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    """
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(
        cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))  # scipy.misc.imread(img_path, mode='RGB'))


def to_numpy(tensor):
    # torch.Tensor => numpy.ndarray
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return np.array(src_result)


def heatmap_to_coord_simple(hms, bbox, **kwargs):
    if not isinstance(hms, np.ndarray):
        hms = hms.cpu().data.numpy()
    coords, maxvals = get_max_pred(hms)

    hm_h = hms.shape[1]
    hm_w = hms.shape[2]

    # post-processing
    for p in range(coords.shape[0]):
        hm = hms[p]
        px = int(round(float(coords[p][0])))
        py = int(round(float(coords[p][1])))
        if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
            diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                             hm[py + 1][px] - hm[py - 1][px]))
            coords[p] += np.sign(diff) * .25

    preds = np.zeros_like(coords)

    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center, scale,
                                   [hm_w, hm_h])

    return preds[None, :, :], maxvals[None, :, :]

def heatmap_to_coord_medical(hms, **kwargs):
    if not isinstance(hms, np.ndarray):
        hms = hms.cpu().data.numpy()
    coords, maxvals = get_max_pred(hms)

    hm_h = hms.shape[1]
    hm_w = hms.shape[2]

    # post-processing
    for p in range(coords.shape[0]):
        hm = hms[p]
        px = int(round(float(coords[p][0])))
        py = int(round(float(coords[p][1])))
        if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
            diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                             hm[py + 1][px] - hm[py - 1][px]))
            coords[p] += np.sign(diff) * .25

    preds = np.zeros_like(coords)

    # transform scale
    w = hm_w * 4
    h = hm_h * 4
    center = np.array([w * 0.5, h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center, scale,
                                   [hm_w, hm_h])

    return preds[None, :, :], maxvals[None, :, :]

def heatmap_to_coord(pred_jts, pred_scores, hm_shape):
    hm_height, hm_width = hm_shape
    hm_height = hm_height * 4
    hm_width = hm_width * 4

    ndims = pred_jts.dim()
    assert ndims in [2, 3], "Dimensions of input heatmap should be 2 or 3"
    if ndims == 2:
        pred_jts = pred_jts.unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0)

    coords = pred_jts.cpu().numpy()
    coords = coords.astype(float)
    pred_scores = pred_scores.cpu().numpy()
    pred_scores = pred_scores.astype(float)

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    preds = np.zeros_like(coords)
    # h, w should be correct
    w = hm_width
    h = hm_height
    center = np.array([w * 0.5, h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            preds[i, j, 0:2] = transform_preds(coords[i, j, 0:2], center, scale,
                                               [hm_width, hm_height])

    return preds, pred_scores




def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.max(heatmaps_reshaped, 1)

    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_max_pred_batch(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0,
                         align=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_warpmatrix(theta, size_input, size_dst, size_target, pixel_std):
    size_target = size_target * pixel_std
    theta = theta / 180.0 * np.pi

    matrix = np.zeros((2, 3), dtype=np.float32)

    scale_x = size_target[0] / size_dst[0]
    scale_y = size_target[1] / size_dst[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = math.sin(theta) * scale_y
    matrix[0, 2] = -0.5 * size_target[0] * math.cos(theta) - 0.5 * size_target[1] * math.sin(theta) + 0.5 * size_input[
        0]
    matrix[1, 0] = -math.sin(theta) * scale_x
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = 0.5 * size_target[0] * math.sin(theta) - 0.5 * size_target[1] * math.cos(theta) + 0.5 * size_input[1]

    return matrix


def get_warpmatrix_inverse(theta, size_input, size_dst, size_target):
    """
    :param theta: angle x y
    :param size_input:[w,h]
    :param size_dst: [w,h] i
    :param size_target: [w,h] b
    :return:
    """
    size_target = size_target * 200.0
    theta = theta / 180.0 * math.pi
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
            -0.5 * size_input[0] * math.cos(theta) + 0.5 * size_input[1] * math.sin(theta) + 0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
            -0.5 * size_input[0] * math.sin(theta) - 0.5 * size_input[1] * math.cos(theta) + 0.5 * size_target[1])
    return matrix


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_func_heatmap_to_coord(cfg):
    if cfg.TEST.get('HEATMAP2COORD') == 'coord':
        return heatmap_to_coord
    elif cfg.TEST.get('HEATMAP2COORD') == 'heatmap':
        return heatmap_to_coord_simple
    else:
        raise NotImplementedError


class get_coord(object):
    def __init__(self, cfg, norm_size):
        self.type = cfg.TEST.get('HEATMAP2COORD')
        self.input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self.norm_size = norm_size

    def __call__(self, output, idx):
        if self.type == 'coord':
            pred_jts = output.pred_pts[idx]
            pred_scores = output.maxvals[idx]
            return heatmap_to_coord(pred_jts, pred_scores, self.norm_size)
        elif self.type == 'heatmap':
            pred_hms = output.heatmap[idx]
            # print('need to correct')
            return heatmap_to_coord_medical(pred_hms)
        else:
            raise NotImplementedError

def rescale_pts(pts, down_ratio):
    return np.asarray(pts, np.float32) / float(down_ratio)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


class ConvertImgFloat(object):
    def __call__(self, img, pts):
        return img.astype(np.float32), pts.astype(np.float32)


class _ConvertImgFloat(object):
    def __call__(self, img, pts):
        return img.astype(np.float32), pts.astype(np.float32)


class RandomContrast(object):
    def __init__(self, lower=0.8, upper=1.2):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, pts):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, pts


class RandomBrightness(object):
    def __init__(self, delta=24):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, pts):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, pts


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img, pts):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, pts


class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        # self.rln = RandomLightingNoise()

    def __call__(self, img, pts):
        img, pts = self.rb(img, pts)
        if random.randint(2):
            distort = self.pd
        else:
            distort = self.pd
        img, pts = distort(img, pts)
        # img, pts = self.rln(img, pts)
        return img, pts


class Expand(object):
    def __init__(self, max_scale=1.5, mean=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, pts):
        if random.randint(2):
            return img, pts
        h, w, c = img.shape
        ratio = random.uniform(1, self.max_scale)
        y1 = random.uniform(0, h * ratio - h)
        x1 = random.uniform(0, w * ratio - w)
        if np.max(pts[:, 0]) + int(x1) > w - 1 or np.max(pts[:, 1]) + int(y1) > h - 1:  # keep all the pts
            return img, pts
        else:
            expand_img = np.zeros(shape=(int(h * ratio), int(w * ratio), c), dtype=img.dtype)
            expand_img[:, :, :] = self.mean
            expand_img[int(y1):int(y1 + h), int(x1):int(x1 + w)] = img
            pts[:, 0] += int(x1)
            pts[:, 1] += int(y1)
            return expand_img, pts


class RandomSampleCrop(object):
    def __init__(self, ratio=(0.5, 1.5), min_win=0.9):
        self.sample_options = (0.7, None)
        # (
        # using entire original input image
        #     None,
        #     # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
        #     # (0.1, None),
        #     # (0.3, None),
        #     (0.7, None),
        #     (0.9, None),
        #     # randomly sample a patch
        #     (None, None),
        # )
        self.ratio = ratio
        self.min_win = min_win

    def __call__(self, img, pts):
        height, width, _ = img.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, pts
            for _ in range(50):
                current_img = img
                current_pts = pts
                w = random.uniform(self.min_win * width, width)
                h = random.uniform(self.min_win * height, height)
                if h / w < self.ratio[0] or h / w > self.ratio[1]:
                    continue
                y1 = random.uniform(height - h)
                x1 = random.uniform(width - w)
                rect = np.array([int(y1), int(x1), int(y1 + h), int(x1 + w)])
                current_img = current_img[rect[0]:rect[2], rect[1]:rect[3], :]
                current_pts[:, 0, 0] -= rect[1]
                current_pts[:, 1, 0] -= rect[0]
                pts_new = []
                for pt in current_pts:
                    # print(pt[0,0])
                    if pt[0, 0] < 0 or pt[1, 0] < 0 or pt[0, 0] > current_img.shape[1] - 1 or pt[1, 0] > \
                            current_img.shape[0] - 1:
                        pt[:, 1] = 0
                    pts_new.append(pt)

                return current_img, np.asarray(pts_new, np.float32)


class RandomMirror_w(object):
    def __call__(self, img, pts):
        _, w, _ = img.shape
        if random.randint(2):
            img = img[:, ::-1, :]
            pts[:, 0] = w - pts[:, 0]
        return img, pts


class RandomMirror_h(object):
    def __call__(self, img, pts):
        h, _, _ = img.shape
        if random.randint(2):
            img = img[::-1, :, :]
            pts[:, 1] = h - pts[:, 1]
        return img, pts


class Resize(object):
    def __init__(self, h, w):
        self.dsize = (w, h)

    def __call__(self, img, pts):
        h, w, c = img.shape
        pts[:, 0] = pts[:, 0] / w * self.dsize[0]
        pts[:, 1] = pts[:, 1] / h * self.dsize[1]
        img = cv2.resize(img, dsize=self.dsize)
        return img, np.asarray(pts)