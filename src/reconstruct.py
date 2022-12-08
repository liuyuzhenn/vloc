"""
File: reconstruct.py
Author: liuyuzhen
Email: liuyuzhen22@mails.ucas.ac.cn
Github: https://github.com/liuyuzhenn
Description: 
"""

import argparse
import torch
import os

from recon.feature import *
from recon.database import import_feature
from recon.match import exhaustive_matching
from recon.model import HyNet, SOSNet


def model_whole_pipline(colmap_path, root_dir, model, threads=2, device='cuda', output_type='TXT'):
    desc_dir = os.path.join(root_dir, 'descriptors')
    kp_dir = os.path.join(root_dir, 'keypoints')
    if not os.path.isdir(desc_dir):
        os.makedirs(desc_dir)
    if not os.path.isdir(kp_dir):
        os.makedirs(kp_dir)

    ################################
    # Extract features
    ################################
    patch_dir = os.path.join(root_dir, 'patches')
    extract_save_model_features(
        patch_dir, root_dir, model, device, 512)

    db_file = os.path.join(root_dir, 'database.db')

    ################################
    # Matching features
    ################################
    exhaustive_matching(root_dir, device=device)

    ################################
    # Import matches to COLMAP
    ################################
    import_feature(root_dir)

    ###############################
    # Verify geometry
    ###############################
    cmd = r'{} matches_importer --database_path {} --match_list_path {} --match_type pairs' \
        .format(colmap_path, db_file, os.path.join(root_dir, 'image-pairs.txt'))
    os.system(cmd)

    ###############################
    # Run COLMAP reconstruction
    ###############################
    sparse = os.path.join(root_dir, 'sparse')
    if not os.path.isdir(sparse):
        os.makedirs(sparse)
    cmd = r'{} mapper --database_path {}  --image_path {} --output_path {} --Mapper.num_threads {} > {}'\
        .format(colmap_path, db_file, root_dir, sparse, threads,
                os.path.join(root_dir, 'log.txt'))
    os.system(cmd)

    ###############################
    # Convert .bin to .txt (or others) for convenient postprocessing
    ###############################
    if output_type != "BIN":
        sparse0 = os.path.join(sparse, '0')
        cmd = r'{}  model_converter --input_path {} --output_type {} --output_path {}'\
            .format(colmap, sparse0, output_type, sparse0)
        os.system(cmd)

        os.remove(os.path.join(sparse0,'cameras.bin'))
        os.remove(os.path.join(sparse0,'images.bin'))
        os.remove(os.path.join(sparse0,'points3D.bin'))


    ###############################
    # Extract result (can be commented)
    ###############################
    cmd = r'{} model_analyzer --path {} > {}'\
        .format(colmap_path, os.path.join(sparse, '0'), os.path.join(root_dir, 'result.txt'))
    os.system(cmd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path', type=str,
                        help='Path to colmap', required=True)
    parser.add_argument('--root_dir', type=str,
                        help='Path to root folder of one scene (see instruction.md)', required=True)
    parser.add_argument('-t', '--threads', type=int,
                        help='Number of threads', default=1)
    # parser.add_argument('-w', '--weight', type=int, default='./weights/HyNet_LIB.pth',
                        # help='Path to model weight')
    parser.add_argument('--device', type=int, default=0,
                        help='-1: CPU; others: GPU')
    parser.add_argument('--desc_type', type=str, default='S',
                        help='S: SOSNet | H:HyNet')
    parser.add_argument('--output_type', type=str, default='TXT',
                        help='BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size when computing decriptors')
    parser.add_argument('--overwrite', type=bool, default=False,
                        help='Whether to overwrite existing files')

    args = parser.parse_args()
    colmap = args.colmap_path
    root_dir = args.root_dir
    threads = args.threads
    device = 'cpu' if args.device < 0 else 'cuda:'+str(args.device)
    # weight = args.weight
    desc = args.desc_type
    output_type = args.output_type
    batch_size = args.batch_size


    img_dir = os.path.join(root_dir,'img')
    out_dir = os.path.join(root_dir, 'patches')

    assert (output_type in ['BIN', 'TXT', 'NVM', 'Bundler', 'VRML', 'PLY', 'R3D', 'CAM'])

    # load model
    cwd = os.path.dirname(__file__)
    if desc=='S':
        model = SOSNet().to(device) # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights','sosnet_32x32_liberty.pth')
    elif desc=='H':
        model = HyNet().to(device) # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights','HyNet_LIB.pth')
    else:
        raise NotImplementedError

    dic = torch.load(weight, map_location=device)
    model.load_state_dict(dic)
    model.eval()

    extract_patches(img_dir, out_dir, bar=True, overwrite=args.overwrite)
    model_whole_pipline(colmap, root_dir, model, threads=threads, device=device, output_type=output_type)

