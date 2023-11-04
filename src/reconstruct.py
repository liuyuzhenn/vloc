import cv2
import numpy as np
import torch
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import argparse
from recon.model import HyNet, SOSNet, ALike, SuperPointModel, configs as alike_cfg
from recon.feature import *
from recon.database import DatabaseOperator
import sys
sys.path.append('./methods/SOLAR/')
from methods.SOLAR.solar_global.utils.networks import load_network
from methods.SuperGluePretrainedNetwork.models.matching import Matching


def main(args):
    desc_type = args.desc_type.lower()
    alike_type = args.alike_type
    img_dir = args.img_dir
    work_space = args.work_space
    prior_focal_length = args.prior_focal_length
    batch_size = args.batch_size
    device = 'cpu' if args.device < 0 else 'cuda:'+str(args.device)
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
    elif desc_type == 'superpoint':
        dic = {
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': args.max_kps,
            'remove_borders': 4,
        }

        model = SuperPointModel(dic).to(device)
    else:
        raise NotImplementedError

    model.eval()

    if not os.path.isdir(work_space):
        os.makedirs(work_space)
    db_file = os.path.join(work_space, 'database.db')
    db = DatabaseOperator(db_file)
    db.create_tables()
    if args.overwrite:
        db.clear_tables()

    # match
    matcher = None
    if args.matcher == 'superglue':
        dic = {
            'superglue': {'weights': args.matcher_weight},
        }
        matcher = Matching(dic).to('cuda')


    if args.matching_mode == 'sequential':
        MODEL = 'resnet101-solar-best.pth'
        IMG_SIZE = 1000

        retrieval_model = load_network(MODEL)
        retrieval_model.eval()
        retrieval_model.cuda()

            # set up the transform
        normalize = transforms.Normalize(
            mean=retrieval_model.meta['mean'],
            std=retrieval_model.meta['std']
        )
        resize = transforms.Resize(IMG_SIZE)


    imgs = os.listdir(img_dir)
    imgs.sort()
    bar = tqdm(range(1, len(imgs)+1), desc='Extract features')
    h, w = None, None
    for image_id in bar:
        ret = db.cursor.execute(
            "SELECT * FROM descriptors WHERE image_id={};".format(image_id)).fetchone()
        if ret is not None:
            continue
        image_name = imgs[image_id-1]

        ##############
        # load image #
        ##############
        img = cv2.imread(os.path.join(img_dir, image_name))
        h, w = img.shape[:2]
    
        if not args.share_intrinsics:
            db.insert_image(image_id, image_name, image_id)
            params = np.array([prior_focal_length*max(h, w), w/2, h/2, 0],
                              dtype=np.float64)  # (f,cx,cy,k)
            camera_model = 2  # 0: simple pinhole, 1: pinhole 2: simple radial
            db.insert_camera(image_id, camera_model, w, h, params.tobytes())
        else:
            db.insert_image(image_id, image_name, 1)
            if image_id==1:
                params = np.array([prior_focal_length*max(h, w), w/2, h/2, 0],
                                  dtype=np.float64)  # (f,cx,cy,k)
                camera_model = 2  # 0: simple pinhole, 1: pinhole 2: simple radial
                db.insert_camera(1, camera_model, w, h, params.tobytes())

        #########################
        # extract local feature #
        #########################
        if desc_type == 'sosnet' or desc_type == 'hynet':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            patches, kps = extract_patches(img, max_kps=args.max_kps)
            if patches.shape[0] > 0:
                descs = extract_desc(model, patches, batch_size, device)
            else:
                descs = np.array([], np.float32)
        elif desc_type == 'alike':
            img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ans = model(img_)
            kps, descs = ans['keypoints'], ans['descriptors']
        elif desc_type == 'superpoint':
            img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
            img_ = torch.from_numpy(img_).float().to(device)
            with torch.no_grad():
                kps, descs, scores = model(img_)

            scores_ = np.stack([scores, np.zeros_like(scores)], axis=1)
            kps = np.concatenate([kps, scores_], axis=1)

        else:
            raise NotImplementedError

        ####################################################
        # extract global feature if using sequential match #
        ####################################################
        if args.matching_mode=='sequential':
            img_ts = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)).cuda().permute(2,0,1)
            img_ts = normalize(img_ts).unsqueeze(0)
            img_ts = resize(img_ts)
            with torch.no_grad():
                desc_g = retrieval_model(img_ts)[0].cpu().numpy()
            db.insert_descriptors_g(image_id, desc_g)

        
        ###########################
        # save data into database #
        ###########################
        db.insert_keypoints(image_id, kps)
        db.insert_descriptors(image_id, descs)
        db.commit()
    
    if args.matching_mode=='sequential':
        del retrieval_model
    

    if args.matching_mode == 'sequential':
        image_pairs = sequential_match_all_desc(
            db.connection, args.overlap, device, matcher=matcher, shape=[h, w], 
            loop_detection_period=args.loop_detection_period,
            loop_detection_num_images=args.loop_detection_num_images)
    elif args.matching_mode == 'exhaustive':
        image_pairs = exhaustive_match_all_desc(
            db.connection, block_size, device, matcher=matcher, shape=[h, w])
    else:
        raise NotImplementedError

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
                        help='Number of threads', default=4)
    parser.add_argument('--device', type=int, default=0,
                        help='-1: CPU; others: GPU')
    parser.add_argument('--output_type', type=str, default='TXT',
                        help='BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM')
    parser.add_argument('--desc_type', type=str, default='alike',
                        help='SOSNet | HyNet | ALIKE | SuperPoint')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size when computing decriptors')
    parser.add_argument('--block_size', type=int, default=50,
                        help='Matching block size')
    parser.add_argument('--prior_focal_length', type=float, default=1.2,
                        help='Prior focal length')
    parser.add_argument('--max_kps', type=int, default=8000,
                        help='Maximum number of keypoints per image')
    parser.add_argument('--scores_th', type=float, default=0.15,
                        help='Detector score threshold (default: 0.15).')
    parser.add_argument('--alike_type', type=str, default='alike-t',
                        help='alike-t | alike-s | alike-n | alike-l')
    parser.add_argument('--matching_mode', type=str, default='exhaustive',
                        help='exhaustive | sequential')
    parser.add_argument('--overlap', type=int, default=20,
                        help='overlap images in exhaustive matching mode')
    parser.add_argument('--overwrite', action='store_true', help='continue')
    parser.add_argument('--share_intrinsics', action='store_true', help='continue')
    parser.add_argument('--matcher', type=str, default='none',
                        help='none | superglue')
    parser.add_argument('--matcher_weight', type=str, default='outdoor',
                        help='outdoor | indoor')
    parser.add_argument('--loop_detection_period', type=int, default=50,
                        help='every N-th image (loop_detection_period) is matched against its visually most similar images (loop_detection_num_images)')
    parser.add_argument('--loop_detection_num_images', type=int, default=20,
                        help='every N-th image (loop_detection_period) is matched against its visually most similar images (loop_detection_num_images)')
    args = parser.parse_args()

    output_type = args.output_type

    assert (output_type in ['BIN', 'TXT', 'NVM',
            'Bundler', 'VRML', 'PLY', 'R3D', 'CAM'])

    main(args)
