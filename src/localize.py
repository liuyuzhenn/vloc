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
from recon.model import HyNet, SOSNet, ALike, configs as alike_cfg


def main(args):
    device = 'cpu' if args.device < 0 else 'cuda:'+str(args.device)
    desc_type = args.desc_type.lower()
    alike_type = args.alike_type
    database = args.database
    folder = args.model_dir
    img = args.img_path
    focal_length = args.focal_length
    ratio = args.ratio
    match_num_kps = args.match_num_kps
    conf = args.confidence
    iters = args.iterations
    error = args.reproj_error
    words = args.visual_words

    cwd = os.path.dirname(__file__)
    ############################################################
    # load descriptor model
    ############################################################
    if desc_type == 'sosnet':
        model_desc = SOSNet().to(device)  # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights', 'sosnet_32x32_liberty.pth')
        dic = torch.load(weight, map_location=device)
        model_desc.load_state_dict(dic)
    elif desc_type == 'hynet':
        model_desc = HyNet().to(device)  # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights', 'HyNet_LIB.pth')
        dic = torch.load(weight, map_location=device)
        model_desc.load_state_dict(dic)
    elif desc_type == 'alike':
        alike_type = args.alike_type
        model_desc = ALike(**alike_cfg[alike_type],
                           device=device,
                           top_k=0,
                           scores_th=args.scores_th,
                           n_limit=args.max_kps)
    else:
        raise NotImplementedError

    model_desc.eval()

    ############################################################
    # load 3D model and clusters
    ############################################################
    model3d = Model3D(folder, database)
    clusters = np.load(words)
    dict_clusterid_points3d = cluster_model3d(clusters, model3d)

    ############################################################
    # load image
    ############################################################
    img = cv2.imread(img)
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
    if desc_type=='sosnet' or desc_type=='hynet':
        patches, kps = extract_patches(img, max_kps=10000)
        descs = extract_desc(model_desc, patches, 512, device)
    elif desc_type=='alike':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ans = model_desc(img)
        kps, descs = ans['keypoints'], ans['descriptors']
    else:
        raise NotImplementedError

    t2 = time.time()

    print('Feature extraction time cost is {:.1f} ms'.format((t2-t1)*1000))

    ############################################################
    # localize
    ############################################################
    t1 = time.time()
    success, rvec, tvec, inliers, corrs_total = localize(kps, descs, camera_matrix, dist_coeff, clusters,
                                            dict_clusterid_points3d, num_kps=match_num_kps, ratio=ratio,
                                            ransac_conf=conf, ransac_iters=iters, ransac_reproj_err=error)
    t2 = time.time()
    print('Localization time cost is {:.1f} ms'.format((t2-t1)*1000))
    print('  ==> status           : {}'.format('success' if success else 'failed'))
    print('  ==> PnP inliers/total: {}/{}'.format(inliers, corrs_total))
    print('  ==> rvec             : {}'.format(rvec.reshape(-1)))
    print('  ==> tvec             : {}'.format(tvec.reshape(-1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ##################################################
    # required
    ##################################################
    parser.add_argument('--img_path', type=str, required=True,
                        help='path to image file')
    parser.add_argument('--database', type=str, required=True,
                        help='path to model database file')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path to the folder containing *.txt files')
    parser.add_argument('--visual_words', type=str, required=True,
                        help='path to the visual words file')

    ##################################################
    # optional
    ##################################################
    parser.add_argument('--desc_type', type=str, default='alike',
                        help='sosnet | hynet | alike')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size when computing decriptors')
    parser.add_argument('--device', type=int, default=0,
                        help='-1: CPU; others: GPU')
    parser.add_argument('--focal_length', type=float, default=1.2,
                        help='focal length of the image')
    parser.add_argument('--match_num_kps', type=int, default=5000,
                        help='Number of keypoints used for matching')
    parser.add_argument('--ratio', type=float, default=0.9,
                        help='ratio test threshold')
    parser.add_argument('--confidence', type=float, default=0.99,
                        help='RanSAC confidence')
    parser.add_argument('--iterations', type=int, default=100,
                        help='RanSAC iterations')
    parser.add_argument('--reproj_error', type=float, default=1.2,
                        help='RanSAC error threshold')
    parser.add_argument('--max_kps', type=int, default=8000,
                        help='Maximum number of features to extract')
    parser.add_argument('--scores_th', type=float, default=0.15,
                        help='Detector score threshold (default: 0.15).')
    parser.add_argument('--alike_type', type=str, default='alike-t',
                        help='Options: alike-t | alike-s | alike-n | alike-l')

    args = parser.parse_args()
    main(args)
