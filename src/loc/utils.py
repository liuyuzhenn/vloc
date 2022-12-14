import numpy as np
import cv2


def compute_distance(a, b):
    distance = np.sum(a**2, axis=1, keepdims=True) + \
        np.sum(b**2, axis=1, keepdims=True).T - 2 * a@b.T
    return distance

def quaternion_to_rotation_matrix(qt):
    quat_v = qt[1:].reshape(3, 1)
    v_ = np.array([
        [0, -qt[3], qt[2]],
        [qt[3], 0, -qt[1]],
        [-qt[2], qt[1], 0]], np.float64)
    R = quat_v @ quat_v.transpose() + \
        qt[0]**2*np.identity(3) + 2*qt[0]*v_ + v_@v_
    return R


def quaternion_to_rvec(qt):
    theta = 2*np.arccos(qt[0])
    n = qt[1:]/np.sin(theta/2)
    return n*theta


def rvec_to_quaternion(rvec):
    theta = np.linalg.norm(rvec)
    q = np.empty((4), np.float64)
    q[0] = np.cos(theta/2)
    q[1:] = rvec.ravel()/theta*np.sin(theta/2)
    return q


def rotation_matrix_to_quaternion(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec_to_quaternion(rvec)
