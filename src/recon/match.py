"""
File: match.py
Author: liuyuzhen
Email: liuyuzhen22@mails.ucas.ac.cn
Github: https://github.com/liuyuzhenn
Description: 
"""

from math import ceil
import torch
import numpy as np
import os


def match_descs(descs1, descs2, device):
    descs1 = torch.from_numpy(descs1.copy()).float().to(device)
    descs2 = torch.from_numpy(descs2.copy()).float().to(device)
    dist_matrix = torch.cdist(descs1, descs2, p=2)
    inds12 = torch.min(dist_matrix, dim=1)[1]
    inds21 = torch.min(dist_matrix, dim=0)[1]
    inds1 = torch.arange(descs1.shape[0], device=device)
    match_ids = torch.stack((inds1, inds12), dim=1)
    mask1 = inds21[inds12] == inds1
    return match_ids[mask1].cpu().numpy()


def load_descriptor(f):
    byte = open(f, 'rb').read()
    nd = np.frombuffer(byte, np.int32, 2, 0)
    descs = np.frombuffer(byte, np.float32, -1, 8).reshape(nd[0], -1)
    return descs


def save_matches(path, matches):
    with open(path, 'wb') as f:
        n = matches.shape[0]
        nd = np.array([n, 2], np.int32)
        f.write(nd)
        byte = matches.astype(np.uint32).tobytes()
        f.write(byte)


def exhaustive_matching(img_dir, block_size=100, device='cuda:0', show_status=True, overwrite=False):
    """block matching to avoid memory overflow"""
    desc_dir = os.path.join(img_dir, 'descriptors')
    matches_dir = os.path.join(img_dir, 'matches')
    if not os.path.isdir(matches_dir):
        os.makedirs(matches_dir)
    files = os.listdir(desc_dir)
    files.sort()
    num_files = len(files)
    num_blocks = ceil(num_files/block_size)
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
                desc_name = files[idx]
                if desc_name not in descs_all:
                    desc = load_descriptor(os.path.join(desc_dir, desc_name))
                    descs_all[desc_name] = desc
            for idx in range(start_idx2, end_idx2):
                desc_name = files[idx]
                if desc_name not in descs_all:
                    desc = load_descriptor(os.path.join(desc_dir, desc_name))
                    descs_all[desc_name] = desc

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
                    desc1_name = files[idx1_]
                    desc2_name = files[idx2_]
                    img1_name = '.'.join(desc1_name.split('.')[:-1])
                    img2_name = '.'.join(desc2_name.split('.')[:-1])
                    path = os.path.join(
                        matches_dir, img1_name+'---'+img2_name+'.bin')
                    if (not overwrite) and os.path.exists(path):
                        continue
                    descs1 = descs_all[desc1_name]
                    descs2 = descs_all[desc2_name]
                    matches = match_descs(descs1, descs2, device)
                    save_matches(path, matches)
