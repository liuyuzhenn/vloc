from tqdm import tqdm
import cv2
import numpy as np
import torch
import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import sqlite3
from recon.model import HyNet, SOSNet
from recon.feature import *


# TODO: time test
def run(model, args):
    img_dir = args.img_dir
    work_space = args.work_space
    prior_focal_length = args.prior_focal_length
    batch_size = args.batch_size
    device = args.device
    block_size = args.block_size
    colmap_path = args.colmap_path
    threads = args.threads

    db_file = os.path.join(work_space, 'database.db')
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    create_tables(cursor)

    cursor.execute("DELETE FROM cameras;")
    cursor.execute("DELETE FROM images;")
    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")

    imgs = os.listdir(img_dir)
    imgs.sort()
    bar = tqdm(range(1, len(imgs)+1), desc='Extract features')
    for image_id in bar:
        image_name = imgs[image_id-1]
        # insert image
        cursor.execute(
            "INSERT INTO images(image_id, name, camera_id, prior_qw, prior_qx, " +
            "prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) VALUES(?, ?, ?, ?, ?, " +
            "?, ?, ?, ?, ?);", (image_id, image_name, image_id, None, None, None, None, None, None, None))

        # get image width, height
        img = cv2.imread(os.path.join(img_dir, image_name),
                         cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        params = np.array([prior_focal_length*max(h, w), w/2, h/2, 0],
                          dtype=np.float64)  # (f,cx,cy,k)
        cam_model = 2  # 0: simple pinhole, 1: pinhole 2: simple radial
        # insert camera
        cursor.execute(
            "INSERT INTO cameras(camera_id, model, width, height, params, " +
            "prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);", (image_id, cam_model, w, h, params.tobytes(), 1))

        # detect keypoints and extract descriptors
        patches, kps = extract_patches(img,max_kps=args.max_kps)
        if patches.shape[0] > 0:
            descs = extract_desc(model, patches, batch_size, device)
        else:
            descs = np.array([], np.float32)

        scale_ori = np.zeros_like(kps)
        kps = np.concatenate((kps, scale_ori), axis=1)  # (n, 4)

        # save keypoints to .db file
        kps_str = kps.tobytes()
        descs_str = descs.tobytes()
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) " +
                       "VALUES(?, ?, ?, ?);",
                       (image_id, kps.shape[0], kps.shape[1],
                        kps_str))
        cursor.execute("INSERT INTO descriptors(image_id, rows, cols, data) " +
                       "VALUES(?, ?, ?, ?);",
                       (image_id, descs.shape[0], descs.shape[1],
                        descs_str))

        connection.commit()

    # match
    image_pairs = match_all_desc(connection, block_size, device)

    with open(os.path.join(work_space, "image-pairs.txt"), "w") as fid:
        for image_name1, image_name2 in image_pairs:
            fid.write("{} {}\n".format(image_name1, image_name2))

    colmap_pipline(colmap_path, img_dir, db_file, work_space, threads)


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


def create_tables(cur):
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS cameras
         (camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
          model                INTEGER                             NOT NULL,
          width                INTEGER                             NOT NULL,
          height               INTEGER                             NOT NULL,
          params               BLOB,
          prior_focal_length   INTEGER                             NOT NULL);
                """)
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS images
         (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
          name       TEXT                                NOT NULL UNIQUE,
          camera_id  INTEGER                             NOT NULL,
          prior_qw   REAL,
          prior_qx   REAL,
          prior_qy   REAL,
          prior_qz   REAL,
          prior_tx   REAL,
          prior_ty   REAL,
          prior_tz   REAL,
      CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 999999),
      FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
      CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);
      """)
    cur.executescript("""
      CREATE TABLE IF NOT EXISTS keypoints
         (image_id  INTEGER  PRIMARY KEY  NOT NULL,
          rows      INTEGER               NOT NULL,
          cols      INTEGER               NOT NULL,
          data      BLOB,
      FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
                """)
    cur.executescript("""
      CREATE TABLE IF NOT EXISTS descriptors
         (image_id  INTEGER  PRIMARY KEY  NOT NULL,
          rows      INTEGER               NOT NULL,
          cols      INTEGER               NOT NULL,
          data      BLOB,
      FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
                """)
    cur.executescript("""
      CREATE TABLE IF NOT EXISTS matches
         (pair_id  INTEGER  PRIMARY KEY  NOT NULL,
          rows     INTEGER               NOT NULL,
          cols     INTEGER               NOT NULL,
          data     BLOB);
                """)
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS two_view_geometries
           (pair_id  INTEGER  PRIMARY KEY  NOT NULL,
            rows     INTEGER               NOT NULL,
            cols     INTEGER               NOT NULL,
            data     BLOB,
            config   INTEGER               NOT NULL,
            F        BLOB,
            E        BLOB,
            H        BLOB,
            qvec     BLOB,
            tvec     BLOB);
                """)


if __name__ == "__main__":
    # parser.add_argument('--colmap_path', type=str, default='E:/software/COLMAP-3.7-windows-cuda/COLMAP.bat',
    # help='Path to colmap')
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path', type=str,
    help='Path to colmap', required=True)
    parser.add_argument('--work_space', type=str, required=True,
                        help='Path to the workspace (see instruction.md)')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Path to the image folder')

    parser.add_argument('-t', '--threads', type=int,
                        help='Number of threads', default=1)
    parser.add_argument('--device', type=int, default=0,
                        help='-1: CPU; others: GPU')
    parser.add_argument('--desc_type', type=str, default='S',
                        help='S: SOSNet | H:HyNet')
    parser.add_argument('--output_type', type=str, default='TXT',
                        help='BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size when computing decriptors')
    parser.add_argument('--block_size', type=int, default=50,
                        help='Matching block size')
    parser.add_argument('--prior_focal_length', type=float, default=1.2,
                        help='Prior focal length')
    parser.add_argument('--max_kps', type=int, default=10000,
                        help='Maximum number of keypoints per image')
    args = parser.parse_args()

    device = 'cpu' if args.device < 0 else 'cuda:'+str(args.device)
    # weight = args.weight
    desc = args.desc_type
    output_type = args.output_type

    assert (output_type in ['BIN', 'TXT', 'NVM',
            'Bundler', 'VRML', 'PLY', 'R3D', 'CAM'])

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

    run(model, args)
