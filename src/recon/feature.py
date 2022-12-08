"""
File: feature.py
Author: liuyuzhen
Email: liuyuzhen22@mails.ucas.ac.cn
Github: https://github.com/liuyuzhenn
Description: 
"""

from math import ceil
import cv2
import numpy as np
import torch
import numpy as np
import cv2


def extract_patches(img, factor=5):
    dst_sz = 32
    c = dst_sz//2
    dst_sz_ = int(dst_sz*1.5)
    c_ = dst_sz_//2

    sift = cv2.SIFT_create()

    h, w = img.shape
    kps = sift.detect(img, None)
    pts = []
    patches = []
    for kp in kps:
        sz = kp.size
        x, y = round(kp.pt[0]), round(kp.pt[1])
        r = round(sz/2*factor*1.5)
        # ignore keypointss that are near the border
        if x-r < 0 or x+r > w or y-r < 0 or y+r > h:
            continue
        # align with the dominant direction
        patch = cv2.resize(img[y-r:y+r, x-r:x+r], (dst_sz_, dst_sz_))
        r = cv2.getRotationMatrix2D((c_, c_), kp.angle, 1)
        patch = cv2.warpAffine(patch, r, (dst_sz_, dst_sz_))[
            c_-c:c_+c, c_-c:c_+c]
        # add img patches and (x,y) coords to the list
        patches.append(patch)
        pts.append(kp.pt)
    patches = np.array(patches)
    pts = np.array(pts, np.float32)
    return patches, pts


@torch.no_grad()
def extract_desc(model, patches, batch_size, device):
    model.eval()
    n_batch = ceil(patches.shape[0]/batch_size)
    features = torch.tensor([])
    for i in range(n_batch):
        batch = torch.from_numpy(
            patches[i*batch_size:min((i+1)*batch_size, len(patches))]).float().to(device)
        batch = batch.unsqueeze(1)
        out = model(batch).cpu()
        features = torch.concat((features, out), axis=0)
    return features.numpy()


def match_all_desc(connection, block_size, device, show_status=True):
    cursor = connection.cursor()
    img_id_name = cursor.execute(
        "SELECT image_id, name FROM images;").fetchall()
    num_files = len(img_id_name)
    num_blocks = ceil(num_files/block_size)
    image_pairs = []
    image_pair_ids = []
    for block_idx1 in range(num_blocks):
        start_idx1 = block_idx1*block_size
        end_idx1 = min((block_idx1+1)*block_size, num_files)
        for block_idx2 in range(num_blocks):
            if show_status:
                print('Matching [{}/{}, {}/{}]'
                      .format(block_idx1+1, num_blocks, block_idx2+1, num_blocks))
            start_idx2 = block_idx2*block_size
            end_idx2 = min((block_idx2+1)*block_size, num_files)

            # Load descriptors
            descs_all = {}
            for idx in range(start_idx1, end_idx1):
                image_id = img_id_name[idx][0]
                if image_id not in descs_all:
                    rows, cols, data = cursor.execute(
                        "SELECT rows,cols,data FROM descriptors WHERE image_id={};".format(image_id)).fetchone()
                    desc = np.frombuffer(
                        data, dtype=np.float32).reshape((rows, cols))
                    descs_all[image_id] = desc

            for idx in range(start_idx2, end_idx2):
                image_id = img_id_name[idx][0]
                if image_id not in descs_all:
                    rows, cols, data = cursor.execute(
                        "SELECT rows,cols,data FROM descriptors WHERE image_id={};".format(image_id)).fetchone()
                    desc = np.frombuffer(
                        data, dtype=np.float32).reshape((rows, cols))
                    descs_all[image_id] = desc

            for idx1 in range(start_idx1, end_idx1):
                for idx2 in range(start_idx2, end_idx2):
                    if idx1 < idx2:
                        idx1_ = idx1
                        idx2_ = idx2
                    elif idx1 > idx2:
                        idx2_ = idx1
                        idx1_ = idx2
                    else:
                        continue

                    image_id1, name1 = img_id_name[idx1_]
                    image_id2, name2 = img_id_name[idx2_]
                    image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
                    if image_pair_id in image_pair_ids:
                        continue
                    else:
                        image_pairs.append((name1, name2))
                        image_pair_ids.append(image_pair_id)

                    descs1 = descs_all[image_id1]
                    descs2 = descs_all[image_id2]

                    matches = match(descs1, descs2, device)
                    matches_str = matches.tobytes()
                    cursor.execute("INSERT INTO  matches(pair_id, rows, cols, data) " +
                                   "VALUES(?, ?, ?, ?);",
                                   (image_pair_id, matches.shape[0], matches.shape[1],
                                    matches_str))
                    connection.commit()
    return image_pairs


def match(descs1, descs2, device):
    descs1 = torch.from_numpy(descs1.copy()).float().to(device)
    descs2 = torch.from_numpy(descs2.copy()).float().to(device)
    dist_matrix = torch.cdist(descs1, descs2, p=2)
    inds12 = torch.min(dist_matrix, dim=1)[1]
    inds21 = torch.min(dist_matrix, dim=0)[1]
    inds1 = torch.arange(descs1.shape[0], device=device)
    match_ids = torch.stack((inds1, inds12), dim=1)
    mask1 = inds21[inds12] == inds1
    return match_ids[mask1].cpu().numpy().astype(np.uint32)


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2

