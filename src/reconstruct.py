import cv2
import numpy as np
import torch
import os
import numpy as np
from tqdm import tqdm
import argparse
from recon.model import HyNet, SOSNet, ALike, configs as alike_cfg
from recon.feature import *
from recon.database import DatabaseOperator


def main(args):
    desc_type = args.desc_type.lower()
    alike_type = args.alike_type
    img_dir = args.img_dir
    work_space = args.work_space
    prior_focal_length = args.prior_focal_length
    batch_size = args.batch_size
    device = 'cpu' if args.device < 0 else 'cuda:'+str(args.device)
    desc_type = args.desc_type
    block_size = args.block_size
    colmap_path = args.colmap_path
    threads = args.threads

    # load model
    cwd = os.path.dirname(__file__)
    if desc_type == 'sosnet':
        model = SOSNet().to(device)  # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights', 'sosnet_32x32_liberty.pth')
        dic = torch.load(weight, map_location=device)
        model.load_state_dict(dic)
    elif desc_type == 'hynet':
        model = HyNet().to(device)  # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights', 'HyNet_LIB.pth')
        dic = torch.load(weight, map_location=device)
        model.load_state_dict(dic)
    elif desc_type == 'alike':
        model = ALike(**alike_cfg[alike_type],
                      device=device,
                      top_k=0,
                      scores_th=args.scores_th,
                      n_limit=args.max_kps)
    else:
        raise NotImplementedError

    model.eval()

    db_file = os.path.join(work_space, 'database.db')
    db = DatabaseOperator(db_file)
    db.create_tables()
    db.clear_tables()

    imgs = os.listdir(img_dir)
    imgs.sort()
    bar = tqdm(range(1, len(imgs)+1), desc='Extract features')
    for image_id in bar:
        image_name = imgs[image_id-1]
        # get image width, height
        img = cv2.imread(os.path.join(img_dir, image_name))
        h, w = img.shape[:2]
        params = np.array([prior_focal_length*max(h, w), w/2, h/2, 0],
                          dtype=np.float64)  # (f,cx,cy,k)
        camera_model = 2  # 0: simple pinhole, 1: pinhole 2: simple radial

        db.insert_image(image_id, image_name, image_id)
        db.insert_camera(image_id, camera_model, w, h, params.tobytes())
        # detect keypoints and extract descriptors

        if desc_type == 'sosnet' or desc_type == 'hynet':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            patches, kps = extract_patches(img, max_kps=args.max_kps)
            if patches.shape[0] > 0:
                descs = extract_desc(model, patches, batch_size, device)
            else:
                descs = np.array([], np.float32)
        elif desc_type == 'alike':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ans = model(img)
            kps, descs = ans['keypoints'], ans['descriptors']
        else:
            raise NotImplementedError

        db.insert_keypoints(image_id, kps)
        db.insert_descriptors(image_id, descs)
        db.commit()

    # match
    image_pairs = match_all_desc(db.connection, block_size, device)

    with open(os.path.join(work_space, "image-pairs.txt"), "w") as fid:
        for image_name1, image_name2 in image_pairs:
            fid.write("{} {}\n".format(image_name1, image_name2))

    colmap_pipline(colmap_path, img_dir, db_file, work_space, threads)
    db.close()


def colmap_pipline(colmap_path, img_dir, db_file, work_space, threads):
    ###############################
    # Verify geometry
    ###############################
    cmd = r'{} matches_importer --database_path {} --match_list_path {} --match_type pairs' \
        .format(colmap_path, db_file, os.path.join(work_space, 'image-pairs.txt'))
    os.system(cmd)

    ###############################
    # Run COLMAP reconstruction
    ###############################
    sparse = os.path.join(work_space, 'sparse')
    if not os.path.isdir(sparse):
        os.makedirs(sparse)
    cmd = r'{} mapper --database_path {}  --image_path {} --output_path {} --Mapper.num_threads {} > {}'\
        .format(colmap_path, db_file, img_dir, sparse, threads,
                os.path.join(work_space, 'log.txt'))
    os.system(cmd)

    ###############################
    # Convert .bin to .txt (or others) for convenient postprocessing
    ###############################
    if output_type != "BIN":
        sparse0 = os.path.join(sparse, '0')
        cmd = r'{}  model_converter --input_path {} --output_type {} --output_path {}'\
            .format(colmap_path, sparse0, output_type, sparse0)
        os.system(cmd)

        os.remove(os.path.join(sparse0, 'cameras.bin'))
        os.remove(os.path.join(sparse0, 'images.bin'))
        os.remove(os.path.join(sparse0, 'points3D.bin'))

    ###############################
    # Extract result (can be commented out)
    ###############################
    cmd = r'{} model_analyzer --path {} > {}'\
        .format(colmap_path, os.path.join(sparse, '0'), os.path.join(work_space, 'result.txt'))
    os.system(cmd)


if __name__ == "__main__":
    cwd = os.path.abspath(__file__)
    parser = argparse.ArgumentParser()
    ##################################################
    # required
    ##################################################
    parser.add_argument('--colmap_path', type=str,
                        help='Path to colmap', required=True)
    parser.add_argument('--work_space', type=str, required=True,
                        help='Path to the workspace (see instruction.md)')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Path to the image folder')

    ##################################################
    # optional
    ##################################################
    parser.add_argument('-t', '--threads', type=int,
                        help='Number of threads', default=1)
    parser.add_argument('--device', type=int, default=0,
                        help='-1: CPU; others: GPU')
    parser.add_argument('--output_type', type=str, default='TXT',
                        help='BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM')
    parser.add_argument('--desc_type', type=str, default='alike',
                        help='SOSNet | HyNet | ALIKE')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size when computing decriptors')
    parser.add_argument('--block_size', type=int, default=50,
                        help='Matching block size')
    parser.add_argument('--prior_focal_length', type=float, default=1.2,
                        help='Prior focal length')
    parser.add_argument('--max_kps', type=int, default=10000,
                        help='Maximum number of keypoints per image')
    parser.add_argument('--scores_th', type=float, default=0.15,
                        help='Detector score threshold (default: 0.15).')
    parser.add_argument('--alike_type', type=str, default='alike-t',
                        help='alike-t | alike-s | alike-n | alike-l')
    args = parser.parse_args()

    output_type = args.output_type

    assert (output_type in ['BIN', 'TXT', 'NVM',
            'Bundler', 'VRML', 'PLY', 'R3D', 'CAM'])

    main(args)
