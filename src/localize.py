"""
File: localize.py
Author: liuyuzhen
Email: liuyuzhen22@mails.ucas.ac.cn
Github: https://github.com/liuyuzhenn
Description: 
"""

import numpy as np
import os
import cv2
import time
import torch
import argparse

from loc.model3d import Model3D
from loc.loc import localize, cluster_model3d
from recon.feature import extract_patches, extract_desc
from recon.model import HyNet, SOSNet


def main(args):
    device = 'cpu' if args.device < 0 else 'cuda:'+str(args.device)
    desc = args.desc_type
    database = args.database
    folder = args.model_dir
    img = args.img_path
    focal_length = args.focal_length
    ratio = args.ratio
    num_kps = args.num_kps
    conf = args.confidence
    iters = args.iterations
    error = args.reproj_error

    cwd = os.path.dirname(__file__)
    ############################################################
    # load descriptor model
    ############################################################
    if desc == 'S':
        model_desc = SOSNet().to(device)  # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights', 'sosnet_32x32_liberty.pth')
    elif desc == 'H':
        model_desc = HyNet().to(device)  # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights', 'HyNet_LIB.pth')
    else:
        raise NotImplementedError

    dic = torch.load(weight, map_location=device)
    model_desc.load_state_dict(dic)
    model_desc.eval()

    ############################################################
    # load 3D model and clusters 
    ############################################################
    model3d = Model3D(folder, database)
    clusters = np.load(os.path.join(cwd, '../data/visualwords.npy'))
    dict_clusterid_points3d = cluster_model3d(clusters, model3d)

    ############################################################
    # load image
    ############################################################
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    f = max(h, w)*focal_length

    camera_matrix = np.array([
        [f, 0., w/2],
        [0., f, h/2],
        [0., 0., 1.],
    ], dtype=np.float32)

    # pass zero if image distortion information is not available
    dist_coeff = np.zeros([4], dtype=np.float32)

    ############################################################
    # extract features
    ############################################################
    t1 = time.time()
    patches, kps = extract_patches(img, max_kps=10000)
    t2 = time.time()
    print('Feature extraction time cost is {:.1f} ms'.format((t2-t1)*1000))
    descs = extract_desc(model_desc, patches, 512, device)

    ############################################################
    # localize
    ############################################################
    t1 = time.time()
    success, rvec, tvec, inliers = localize(kps, descs, camera_matrix, dist_coeff, clusters,
                                            dict_clusterid_points3d, num_kps=num_kps, ratio=ratio,
                                            ransac_conf=conf, ransac_iters=iters, ransac_reproj_err=error)
    t2 = time.time()
    print('Localization time cost is {:.1f} ms'.format((t2-t1)*1000))
    print('  ==> status: {}'.format('success' if success else 'failed'))
    print('  ==> inliers: {}'.format(inliers))
    print('  ==> rvec: {}'.format(rvec.reshape(-1)))
    print('  ==> tvec: {}'.format(tvec.reshape(-1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True,
                        help='path to image file')
    parser.add_argument('--database', type=str, required=True,
                        help='path to model database file')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path to the folder containing *.txt files')
    parser.add_argument('--desc_type', type=str, default='S',
                        help='S: SOSNet | H:HyNet.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size when computing decriptors')
    parser.add_argument('--device', type=int, default=0,
                        help='-1: CPU; others: GPU')
    parser.add_argument('--focal_length', type=float, default=1.2,
                        help='focal length of the image')
    parser.add_argument('--num_kps', type=int, default=5000,
                        help='number of keypoints used for PnP')
    parser.add_argument('--ratio', type=float, default=0.9,
                        help='ratio test threshold')
    parser.add_argument('--confidence', type=float, default=0.99,
                        help='RanSAC confidence')
    parser.add_argument('--iterations', type=int, default=100,
                        help='RanSAC iterations')
    parser.add_argument('--reproj_error', type=float, default=1.2,
                        help='RanSAC error threshold')

    args = parser.parse_args()
    main(args)
