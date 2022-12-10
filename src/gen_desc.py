import cv2
from tqdm import tqdm
import argparse
import numpy as np
import torch
import os
from recon.database import DatabaseOperator
from recon.model import HyNet, SOSNet
from recon.feature import *
import time

def main(model, args):
    batch_size = args.batch_size
    db_file = args.database
    img_list_file = args.img_list

    db = DatabaseOperator(db_file)

    db.create_tables()
    db.clear_tables()

    with open(img_list_file, 'r') as f:
        img_list = f.readlines()
        img_list = [img.replace('\n', '') for img in img_list]
    time_kps = 0
    time_desc = 0
    bar = tqdm(range(1, len(img_list) + 1), desc='Extract features')
    for image_id in bar:
        image_name = img_list[image_id-1]

        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        params = np.array([1.2*max(h, w), w/2, h/2, 0],
                          dtype=np.float64)  # (f,cx,cy,k)

        db.insert_image(image_id, image_name, image_id)
        db.insert_camera(image_id, 2, w, h, params.tobytes())

        # detect keypoints and extract descriptors
        t0 =time.time()
        patches, kps = extract_patches(img, max_kps=args.max_kps)
        t1 =time.time()
        if patches.shape[0] > 0:
            descs = extract_desc(model, patches, batch_size, device)
        else:
            descs = np.array([], np.float32)
        t2 =time.time()
        time_kps += t1-t0
        time_desc += t2-t1

        db.insert_keypoints(image_id, kps)
        db.insert_descriptors(image_id, descs)
        db.commit()
    db.close()
    print('kps: {:.3f}'.format(time_kps))
    print('desc: {:.3f}'.format(time_desc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--img_list', type=str, required=True)
    parser.add_argument('--desc_type', type=str, default='S',
                        help='S: SOSNet | H:HyNet')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size when computing decriptors')
    parser.add_argument('--max_kps', type=int, default=10000,
                        help='Maximum number of keypoints per image')
    parser.add_argument('--device', type=int, default=0,
                        help='-1: CPU; others: GPU')
    args = parser.parse_args()

    device = 'cpu' if args.device < 0 else 'cuda:'+str(args.device)
    desc = args.desc_type

    # load model
    cwd = os.path.dirname(__file__)
    if desc == 'S':
        model = SOSNet().to(device)  # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights', 'sosnet_32x32_liberty.pth')
    elif desc == 'H':
        model = HyNet().to(device)  # SOSNet, HyNet
        weight = os.path.join(cwd, '../weights', 'HyNet_LIB.pth')
    else:
        raise NotImplementedError

    dic = torch.load(weight, map_location=device)
    model.load_state_dict(dic)
    model.eval()

    main(model, args)
