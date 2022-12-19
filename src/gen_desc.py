import cv2
from tqdm import tqdm
import argparse
import numpy as np
import torch
import os
from recon.database import DatabaseOperator
from recon.model import HyNet, SOSNet, ALike, configs as alike_cfg
from recon.feature import *
import time


def main(args):
    batch_size = args.batch_size
    db_file = args.database
    img_list_file = args.img_list
    desc_type = args.desc_type
    alike_type = args.alike_type

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

    db = DatabaseOperator(db_file)

    db.create_tables()
    db.clear_tables() # clear previous table

    with open(img_list_file, 'r') as f:
        img_list = f.readlines()
        img_list = [img.strip('\n') for img in img_list if img != '\n']
    bar = tqdm(range(1, len(img_list) + 1), desc='Extract features')
    num_desc = 0
    t1 = time.time()
    for image_id in bar:
        image_name = img_list[image_id-1]

        img = cv2.imread(image_name)
        if img is None:
            print('Failed to load {}'.format(image_name))
            continue
        h, w = img.shape[:2]
        params = np.array([1.2*max(h, w), w/2, h/2, 0],
                          dtype=np.float64)  # (f,cx,cy,k)

        db.insert_image(image_id, image_name, image_id)
        db.insert_camera(image_id, 2, w, h, params.tobytes())

        # detect keypoints and extract descriptors
        if desc_type=='sosnet' or desc_type=='hynet':
            patches, kps = extract_patches(img, max_kps=10000)
            descs = extract_desc(model, patches, 512, device)
        elif desc_type=='alike':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ans = model(img)
            kps, descs = ans['keypoints'], ans['descriptors']
        else:
            raise NotImplementedError
        num_desc += len(desc)

        db.insert_keypoints(image_id, kps)
        db.insert_descriptors(image_id, descs)
        db.commit()
    t2 = time.time()
    db.close()
    print('Time cost is {:.1f}s'.format(t2-t1))
    print('Total descriptros: {}'.format(num_desc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ##################################################
    # required
    ##################################################
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--img_list', type=str, required=True)
    ##################################################
    # optional
    ##################################################
    parser.add_argument('--desc_type', type=str, default='alike',
                        help='SOSNet | HyNet | ALIKE')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size when computing decriptors')
    parser.add_argument('--max_kps', type=int, default=10000,
                        help='Maximum number of keypoints per image')
    parser.add_argument('--device', type=int, default=0,
                        help='-1: CPU; others: GPU')
    parser.add_argument('--scores_th', type=float, default=0.15,
                        help='Detector score threshold (default: 0.15).')
    parser.add_argument('--alike_type', type=str, default='alike-t',
                        help='alike-t | alike-s | alike-n | alike-l')
    args = parser.parse_args()

    device = 'cpu' if args.device < 0 else 'cuda:'+str(args.device)
    desc = args.desc_type

    # load model
    cwd = os.path.dirname(__file__)

    main(args)
