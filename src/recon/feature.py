"""
File: feature.py
Author: liuyuzhen
Email: liuyuzhen22@mails.ucas.ac.cn
Github: https://github.com/liuyuzhenn
Description: 
"""

from math import ceil
from tqdm import tqdm
import cv2
import numpy as np
import torch
import os
import numpy as np
import cv2
from tqdm import tqdm


def extract_patches(img_dir, out_dir, bar=True, factor=5, overwrite=False):
    dst_sz = 32
    c = dst_sz//2
    dst_sz_ = int(dst_sz*1.5)
    c_ = dst_sz_//2

    sift = cv2.SIFT_create()
    images = os.listdir(img_dir)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # show status bar
    if bar:
        bar = tqdm(images, desc='Detecting keypoints')
    else:
        bar = images

    for img in bar:
        file_path = os.path.join(out_dir, img+'.npz')
        if (not overwrite) and os.path.exists(file_path):
            continue
        p = os.path.join(img_dir, img)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
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
        # save as *.npz file, where * is the image name,
        np.savez(file_path, patches=patches, points=pts)


def extract_sift_features(patches, patch_size=32):
    c = patch_size/2.0
    center_kp = cv2.KeyPoint()
    center_kp.pt = (c, c)
    center_kp.size = patch_size/5.303
    sift = cv2.SIFT_create()
    features = np.array([sift.compute(p, [center_kp])[1][0] for p in patches])
    return features


@torch.no_grad()
def extract_model_features(model, patches, device, batch_size=1024):
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


def save_features(out_dir, descriptors, keypoints):
    desc_dir = os.path.join(out_dir, 'descriptors')
    kp_dir = os.path.join(out_dir, 'keypoints')

    if not os.path.isdir(desc_dir):
        os.makedirs(desc_dir)
    if not os.path.isdir(kp_dir):
        os.makedirs(kp_dir)

    for name in descriptors.keys():
        descs = descriptors[name]
        kps = keypoints[name]
        n = descs.shape[0]
        desc_file = os.path.join(desc_dir, name+'.bin')
        kp_file = os.path.join(kp_dir, name+'.bin')
        if os.path.exists(desc_file) and os.path.exists(kp_file):
            continue

        with open(desc_file, 'wb') as f:
            if n > 0:
                nd = np.array([n, descs.shape[1]], dtype=np.int32)
            else:
                nd = np.array([0, 128], dtype=np.int32)
            f.write(nd)
            f.write(descs.tobytes())
        with open(kp_file, 'wb') as f:
            nd = np.array([n, 4], dtype=np.int32)
            f.write(nd)
            scale_ori = np.zeros_like(kps)
            kps = np.concatenate((kps, scale_ori), axis=1)
            f.write(kps.tobytes())


def save_descriptor(path, descs):
    n = descs.shape[0]
    with open(path, 'wb') as f:
        if n > 0:
            nd = np.array([n, 128], dtype=np.int32)
        else:
            nd = np.array([0, 128], dtype=np.int32)
        f.write(nd.tobytes())
        f.write(descs.tobytes())


def save_kps(path, kps):
    n = kps.shape[0]
    with open(path, 'wb') as f:
        nd = np.array([n, 4], dtype=np.int32)
        f.write(nd.tobytes())
        if n == 0:
            return
        scale_ori = np.zeros_like(kps)
        kps = np.concatenate((kps, scale_ori), axis=1)
        f.write(kps.tobytes())


def extract_save_sift_features(patch_dir, root_dir):
    desc_dir = os.path.join(root_dir, 'descriptors')
    kp_dir = os.path.join(root_dir, 'keypoints')
    files = os.listdir(patch_dir)
    for f in tqdm(files, desc='Extracting features'):
        data = np.load(os.path.join(patch_dir, f))
        img_name = '.'.join(f.split('.')[:-1])
        p = data['patches']
        kps = data['points']
        kps_path = os.path.join(kp_dir, img_name+'.bin')
        desc_path = os.path.join(desc_dir, img_name+'.bin')
        if not os.path.exists(desc_path):
            descs = extract_sift_features(p)
            save_descriptor(desc_path, descs)
        if not os.path.exists(kps_path):
            save_kps(kps_path, kps)


def extract_save_model_features(patch_dir, root_dir, model, device, batch_size=512, overwrite=False):
    desc_dir = os.path.join(root_dir, 'descriptors')
    kp_dir = os.path.join(root_dir, 'keypoints')
    files = os.listdir(patch_dir)
    for f in tqdm(files, desc='Extracting features'):
        data = np.load(os.path.join(patch_dir, f))
        img_name = '.'.join(f.split('.')[:-1])
        p = data['patches']
        kps = data['points']
        kps_path = os.path.join(kp_dir, img_name+'.bin')
        desc_path = os.path.join(desc_dir, img_name+'.bin')
        
        # skip existing files if overwrite is False
        if not overwrite and os.path.exists(desc_path):
            continue

        if p.shape[0] > 0:
            descs = extract_model_features(model, p, device, batch_size)
        else:
            descs = np.array([], np.float32)

        save_descriptor(desc_path, descs)

        if not overwrite and os.path.exists(kps_path):
            continue
        save_kps(kps_path, kps)


if __name__ == '__main__':
    pass
