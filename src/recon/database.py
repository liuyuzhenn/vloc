"""
File: import_feature.py
Author: liuyuzhen
Email: liuyuzhen22@mails.ucas.ac.cn
Github: https://github.com/liuyuzhenn
Description: 
"""

import os
import glob
import argparse
import sqlite3
import numpy as np
import cv2


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--feature_path", required=True)
    args = parser.parse_args()
    return args


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def read_matrix(path, dtype):
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, count=2, dtype=np.int32)
        matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)


def import_feature(root_dir):
    # src_file = os.path.join(feature_path, "database.db")
    db_file = os.path.join(root_dir, 'database.db')
    # shutil.copyfile(src_file, dst_file)
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    create_tables(cursor)

    cursor.execute("DELETE FROM cameras;")
    cursor.execute("DELETE FROM images;")

    img_dir = os.path.join(root_dir, 'img')
    imgs = os.listdir(img_dir)
    for i, image_name in enumerate(imgs, 1):
        # insert image
        cursor.execute(
            "INSERT INTO images(image_id, name, camera_id, prior_qw, prior_qx, " +
            "prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) VALUES(?, ?, ?, ?, ?, " +
            "?, ?, ?, ?, ?);", (i, image_name, 1, None, None, None, None, None, None, None))

        # get image width, height
        img = cv2.imread(os.path.join(img_dir, image_name))
        h, w = img.shape[:2]
        # set initial focal length to 1.2*max(h,w)
        params = np.array([1.2*max(h, w), w/2, h/2, 0],
                          dtype=np.float64)  # (f,cx,cy,k)
        cam_model = 2  # 0: simple pinhole, 1: pinhole 2: simple radial
        # insert camera
        cursor.execute(
            "INSERT INTO cameras(camera_id, model, width, height, params, " +
            "prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);", (i, cam_model, w, h, params.tobytes(), 1))

    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    try:
        cursor.execute("DELETE FROM two_view_geometries;")
    except:
        pass
    connection.commit()

    images = {}
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]

    for image_name, image_id in images.items():
        print("Importing features for", image_name)
        keypoint_path = os.path.join(root_dir, "keypoints",
                                     image_name + ".bin")
        keypoints = read_matrix(keypoint_path, np.float32)
        keypoints[:, :2] += 0.5
        descriptor_path = os.path.join(root_dir, "descriptors",
                                       image_name + ".bin")
        descriptors = read_matrix(descriptor_path, np.float32)
        assert keypoints.shape[1] == 4
        assert keypoints.shape[0] == descriptors.shape[0]
        keypoints_str = keypoints.tobytes()
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) " +
                       "VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1],
                        keypoints_str))
        connection.commit()

    image_pairs = []
    for match_path in glob.glob(os.path.join(root_dir,
                                             "matches/*---*.bin")):
        image_name1, image_name2 = \
            os.path.basename(match_path[:-4]).split("---")
        image_pairs.append((image_name1, image_name2))
        print("Importing matches for", image_name1, "---", image_name2)
        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = read_matrix(match_path, np.uint32)
        assert matches.shape[1] == 2
        # if IS_PYTHON3:
        matches_str = matches.tobytes()
        # else:
        #     matches_str = np.getbuffer(matches)
        cursor.execute("INSERT INTO  matches(pair_id, rows, cols, data) " +
                       "VALUES(?, ?, ?, ?);",
                       (image_pair_id, matches.shape[0], matches.shape[1],
                        matches_str))
        connection.commit()

    with open(os.path.join(root_dir, "image-pairs.txt"), "w") as fid:
        for image_name1, image_name2 in image_pairs:
            fid.write("{} {}\n".format(image_name1, image_name2))
    cursor.close()
    connection.close()


if __name__ == "__main__":
    pass
