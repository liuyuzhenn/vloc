import numpy as np
import torch
import cv2
from loc.model3d import Model3D
from loc.utils import *


def cluster_model3d(clusters: np.ndarray, model3d: Model3D) -> dict:
    """
    Returns
    ---
    A dict whose key is the cluster id, and value is a tuple of the form (points3D: list, descriptors: list)

    """
    k, _ = clusters.shape[:2]
    # (n, dim)
    points3d = model3d.points3d.values()
    descriptors = np.stack([p.descriptor for p in points3d], axis=0)
    xyz = np.stack([p.point3d for p in points3d], axis=0)

    distance = compute_distance(descriptors, clusters)
    inds = np.argmin(distance, 1)
    clustered_pointsid = [np.argwhere(inds == i).reshape(-1) for i in range(k)]

    res = {i: (xyz[ids], descriptors[ids])
           for i, ids in enumerate(clustered_pointsid)}
    return res


def localize(keypoints: torch.tensor, descriptors: torch.tensor,
             camera_matrix: np.ndarray, dist_coeff: np.ndarray,
             clusters: np.ndarray, dict_clusterid_points3d: dict,
             num_kps: int = 1000, ratio: float = 0.9,
             ransac_reproj_err: float = 8.0, ransac_conf: float = 0.999, ransac_iters: int = 500) -> tuple():
    """
    Params
    ---
    keypoints: (n, 2) 2d points    
    descriptors: (n, dim) descriptors
    camera_matrix: (3, 3) camera matrix
    dist_coeff: distortion coefficient(https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e)
    clusters: (k, dim) clusters centers
    dict_clusterid_points3d: dict(key=clusterid, value=tuple(points3d, descritpors))
    num_kps: number of keypoints with the lowest matching cost for PnP
    ratio: ratio test threshold
    ransac_reproj_err: reprojection error threshold in RanSAC
    ransac_conf: confidence of RanSAC
    ransac_iters: maximum iterations of RanSAC
    """
    n, dim = descriptors.shape

    # the main time cost comes from here:
    # find the corresponding visual words for each 2D keypoints
    # distance = compute_distance(descriptors, clusters)
    if type(descriptors)==np.ndarray:
        descriptors = torch.from_numpy(descriptors)
    if type(clusters)==np.ndarray:
        clusters = torch.from_numpy(clusters)
    
    distance = torch.cdist(descriptors, clusters)
    inds = torch.argmin(distance, axis=1).numpy()
    
    # convert back to numpy for knn math in cv2
    descriptors = descriptors.numpy()

    # retrieve all 3D points in the same visual word
    # (kp_id, points3d, descriptors)
    metas = [(i, dict_clusterid_points3d[inds[i]][0],
              dict_clusterid_points3d[inds[i]][1]) for i in range(n)]
    metas.sort(key=lambda x: len(x[1]), reverse=False)

    # remove elements that contain zero descriptors
    metas = [p for p in metas if len(p[1]) > 0]
    metas = metas[:num_kps]

    matcher = cv2.BFMatcher()

    obj_points = []
    img_points = []
    # match 2D-3D pairs using ratio test
    for meta in metas:
        kp_id, points3d, descs_3d = meta
        matches = matcher.knnMatch(
            descriptors[kp_id].reshape((1, dim)), descs_3d, k=2)[0]
        if len(matches) == 1 or matches[0].distance <= matches[1].distance*ratio:
            idx = matches[0].trainIdx
            obj_points.append(points3d[idx])
            img_points.append(keypoints[kp_id])

    # convert into numnp array
    obj_points = np.array(obj_points)
    img_points = np.array(img_points)

    # PnP with Ransac
    success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points.reshape(-1, 1, 3), img_points.reshape(-1, 1, 2),
                                                      camera_matrix, dist_coeff, reprojectionError=ransac_reproj_err,
                                                      confidence=ransac_conf, iterationsCount=ransac_iters)
    # R = cv2.Rodrigues(rvec) # angle axis -> rotation matrix

    return success, rvec, tvec, 0 if not success else len(inliers), len(obj_points)
