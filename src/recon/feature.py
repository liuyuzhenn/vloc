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
import time
import cv2
from loc.utils import compute_distance



def extract_patches(img, factor=5, max_kps=10000):
    dst_sz = 32
    c = dst_sz//2
    dst_sz_ = int(dst_sz*1.5)
    c_ = dst_sz_//2

    sift = cv2.SIFT_create(nfeatures=max_kps)

    h, w = img.shape[:2]
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


def exhaustive_match_all_desc(connection, block_size, device, show_status=True, overwrite=False, matcher=None, **kwargs):
    cursor = connection.cursor()
    if overwrite:
        cursor.execute("DELETE FROM matches;")
    img_id_name = cursor.execute(
        "SELECT image_id, name FROM images;").fetchall()
    num_files = len(img_id_name)
    num_blocks = ceil(num_files/block_size)
    image_pairs = []
    t1 = time.time()
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
            kps_all = {}
            for idx in range(start_idx1, end_idx1):
                image_id = img_id_name[idx][0]
                if image_id not in descs_all:
                    rows, cols, data = cursor.execute(
                        "SELECT rows,cols,data FROM descriptors WHERE image_id={};".format(image_id)).fetchone()
                    desc = np.frombuffer(
                        data, dtype=np.float32).reshape((rows, cols))
                    descs_all[image_id] = desc

                    rows, cols, data = cursor.execute(
                        "SELECT rows,cols,data FROM keypoints WHERE image_id={};".format(image_id)).fetchone()
                    kps = np.frombuffer(
                        data, dtype=np.float32).reshape((rows, cols))
                    kps_all[image_id] = kps

            for idx in range(start_idx2, end_idx2):
                image_id = img_id_name[idx][0]
                if image_id not in descs_all:
                    rows, cols, data = cursor.execute(
                        "SELECT rows,cols,data FROM descriptors WHERE image_id={};".format(image_id)).fetchone()
                    desc = np.frombuffer(
                        data, dtype=np.float32).reshape((rows, cols))
                    descs_all[image_id] = desc

                    rows, cols, data = cursor.execute(
                        "SELECT rows,cols,data FROM keypoints WHERE image_id={};".format(image_id)).fetchone()
                    kps = np.frombuffer(
                        data, dtype=np.float32).reshape((rows, cols))
                    kps_all[image_id] = kps

            for idx1 in range(start_idx1, end_idx1):
                for idx2 in range(start_idx2, end_idx2):
                    i = idx1 - start_idx1
                    j = idx2 - start_idx2
                    if block_idx1 <= block_idx2:  # above the block diagonal
                        if i <= j:
                            continue
                    elif i < j:  # below the block diagonal
                        continue

                    if idx1 > idx2:
                        idx1_, idx2_ = idx2, idx1
                    else:
                        idx1_, idx2_ = idx1, idx2

                    image_id1, name1 = img_id_name[idx1_]
                    image_id2, name2 = img_id_name[idx2_]
                    image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
                    image_pairs.append((name1, name2))
                    ret = cursor.execute(
                        "SELECT * FROM matches WHERE pair_id={};".format(image_pair_id)).fetchone()
                    if ret is not None:
                        continue

                    descs1 = descs_all[image_id1]
                    descs2 = descs_all[image_id2]

                    matches = match(
                        descs1, kps_all[image_id1], descs2, kps_all[image_id2], device, matcher, **kwargs)
                    matches_str = matches.tobytes()
                    cursor.execute("INSERT INTO  matches(pair_id, rows, cols, data) " +
                                   "VALUES(?, ?, ?, ?);",
                                   (image_pair_id, matches.shape[0], matches.shape[1],
                                    matches_str))
            connection.commit()
    t2 = time.time()
    print('Matching time cost is {:.4f}s'.format(t2-t1))
    return image_pairs


def sequential_match_all_desc(connection, overlap, device, show_status=True, overwrite=False, matcher=None, **kwargs):
    cursor = connection.cursor()
    if overwrite:
        cursor.execute("DELETE FROM matches;")
    connection.commit()
    img_id_name = cursor.execute(
        "SELECT image_id, name FROM images;").fetchall()
    num_images = len(img_id_name)
    image_pairs = []
    t1 = time.time()

    # Load descriptors
    descs_all = {}
    kps_all = {}
    for idx1 in range(num_images):
        if show_status:
            print('Matching [{}/{}]'.format(idx1+1, num_images))
        image_id1, name1 = img_id_name[idx1]

        if image_id1 not in descs_all:
            rows, cols, data = cursor.execute(
                "SELECT rows,cols,data FROM descriptors WHERE image_id={};".format(image_id1)).fetchone()
            desc = np.frombuffer(
                data, dtype=np.float32).reshape((rows, cols))
            descs_all[image_id1] = desc

            rows, cols, data = cursor.execute(
                "SELECT rows,cols,data FROM keypoints WHERE image_id={};".format(image_id1)).fetchone()
            kps = np.frombuffer(
                data, dtype=np.float32).reshape((rows, cols))
            kps_all[image_id1] = kps

        descs1 = descs_all[image_id1]
        for idx2 in range(idx1+1, idx1+overlap+1):
            if idx2 >= num_images:
                break

            image_id2, name2 = img_id_name[idx2]

            image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
            image_pairs.append((name1, name2))
            ret = cursor.execute(
                "SELECT * FROM matches WHERE pair_id={};".format(image_pair_id)).fetchone()
            if ret is not None:
                continue

            if image_id2 not in descs_all:
                rows, cols, data = cursor.execute(
                    "SELECT rows,cols,data FROM descriptors WHERE image_id={};".format(image_id2)).fetchone()
                desc = np.frombuffer(
                    data, dtype=np.float32).reshape((rows, cols))
                descs_all[image_id2] = desc

                rows, cols, data = cursor.execute(
                    "SELECT rows,cols,data FROM keypoints WHERE image_id={};".format(image_id2)).fetchone()
                kps = np.frombuffer(
                    data, dtype=np.float32).reshape((rows, cols))
                kps_all[image_id2] = kps

            descs1 = descs_all[image_id1]
            descs2 = descs_all[image_id2]
            kps1 = kps_all[image_id1]
            kps2 = kps_all[image_id2]
            matches = match(descs1, kps1, descs2, kps2,
                            device, matcher, **kwargs)
            matches_str = matches.tobytes()
            cursor.execute("INSERT INTO  matches(pair_id, rows, cols, data) " +
                           "VALUES(?, ?, ?, ?);",
                           (image_pair_id, matches.shape[0], matches.shape[1],
                            matches_str))
        connection.commit()
        
    #######################################
    # loop detection with image retrieval #
    #######################################
    if show_status:
        print('Loop detection...')
    
    # extract image retrieval descriptors
    loop_detection_period = kwargs['loop_detection_period']
    loop_detection_num_images = kwargs['loop_detection_num_images']
    descs_g_all = []
    for idx in range(num_images):
        image_id, name = img_id_name[idx]
        dim, data = cursor.execute(
            "SELECT dim,data FROM descriptors_g WHERE image_id={};".format(image_id)).fetchone()
        desc = np.frombuffer(
            data, dtype=np.float32).reshape(dim)
        descs_g_all.append(desc)

    descs_g_all = np.stack(descs_g_all)
    dist = compute_distance(descs_g_all, descs_g_all)
    lis = list(range(0, num_images, loop_detection_period))
    for i in range(len(lis)):
        idx1 = lis[i]
        image_id1, name1 = img_id_name[idx1]
        if show_status:
            print('Matching [{}/{}]'.format(i+1, len(lis)))
        dist_row = dist[idx1]
        inds_retrieved = np.argsort(dist_row)
        for idx2 in inds_retrieved[1:loop_detection_num_images+1]:
            image_id2, name2 = img_id_name[idx2]
            image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
            if image_pair_id not in image_pairs:
                image_pairs.append((name1, name2))
            ret = cursor.execute(
                "SELECT * FROM matches WHERE pair_id={};".format(image_pair_id)).fetchone()
            if ret is not None:
                continue
            descs1 = descs_all[image_id1]
            descs2 = descs_all[image_id2]
            kps1 = kps_all[image_id1]
            kps2 = kps_all[image_id2]
            matches = match(descs1, kps1, descs2, kps2,
                            device, matcher, **kwargs)
            matches_str = matches.tobytes()
            cursor.execute("INSERT INTO  matches(pair_id, rows, cols, data) " +
                           "VALUES(?, ?, ?, ?);",
                           (image_pair_id, matches.shape[0], matches.shape[1],
                            matches_str))
        connection.commit()
    t2 = time.time()
    print('Matching time cost is {:.4f}s'.format(t2-t1))
    return image_pairs


def match(descs1, kps1, descs2, kps2, device, matcher, **kwargs):
    if matcher is None:
        descs1 = torch.from_numpy(descs1.copy()).float().to(device)
        descs2 = torch.from_numpy(descs2.copy()).float().to(device)
        dist_matrix = torch.cdist(descs1, descs2, p=2)
        inds12 = torch.min(dist_matrix, dim=1)[1]
        inds21 = torch.min(dist_matrix, dim=0)[1]
        inds1 = torch.arange(descs1.shape[0], device=device)
        match_ids = torch.stack((inds1, inds12), dim=1)
        mask1 = inds21[inds12] == inds1
        return match_ids[mask1].cpu().numpy().astype(np.uint32)
    else:
        seudo_img = torch.zeros(kwargs['shape']).unsqueeze(0).unsqueeze(0).to('cuda')
        pts1 = kps1[:, :2]
        pts2 = kps2[:, :2]
        scores1 = kps1[:, 2]
        scores2 = kps2[:, 2]
        descs1 = torch.from_numpy(descs1.copy()).float().to(device).unsqueeze(0)
        descs2 = torch.from_numpy(descs2.copy()).float().to(device).unsqueeze(0)
        pts1 = torch.from_numpy(pts1.copy()).float().to(device).unsqueeze(0)
        pts2 = torch.from_numpy(pts2.copy()).float().to(device).unsqueeze(0)
        scores1 = torch.from_numpy(scores1.copy()).float().to(device).unsqueeze(0)
        scores2 = torch.from_numpy(scores2.copy()).float().to(device).unsqueeze(0)
        data = {
            'image0': seudo_img,
            'image1': seudo_img,
            'keypoints0': pts1,
            'keypoints1': pts2,
            'descriptors0': descs1.transpose(-2, -1),
            'descriptors1': descs2.transpose(-2, -1),
            'scores0': scores1,
            'scores1': scores2,
        }
        with torch.no_grad():
            pred = matcher(data)
        matches = pred['matches0'][0].cpu().numpy()
        valid = matches > -1
        idx = np.arange(len(matches))
        matches_ret = np.stack([idx[valid], matches[valid]], axis=1).astype(np.uint32)
        return matches_ret


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2
