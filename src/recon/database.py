"""
File: database.py
Author: liuyuzhen
Email: liuyuzhen22@mails.ucas.ac.cn
Github: https://github.com/liuyuzhenn
Description: 
"""

import sqlite3
import numpy as np

class DatabaseOperator(object):
    """
    Used for database operation, including:
        - create/clear table
        - insert/query image/camera/descriptor/keypoints

    **Note**
        remember to call commit() after insert/delete
    """

    def __init__(self, db_file):
        """db_file: path to *.db. """
        self.db_file = db_file
        self.connection = sqlite3.connect(db_file)
        self.cursor = self.connection.cursor()

    def clear_tables(self):
        """Set all tables to empty
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM cameras;")
        cursor.execute("DELETE FROM images;")
        cursor.execute("DELETE FROM keypoints;")
        cursor.execute("DELETE FROM descriptors;")
        cursor.execute("DELETE FROM matches;")
        cursor.execute("DELETE FROM two_view_geometries;")
        return True

    def create_tables(self):
        """Create all table used in colmap:
            - cameras
            - images
            - keypoints
            - descriptors
            - matches
            - two_view_geometries

        """
        cursor = self.connection.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS cameras
             (camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
              model                INTEGER                             NOT NULL,
              width                INTEGER                             NOT NULL,
              height               INTEGER                             NOT NULL,
              params               BLOB,
              prior_focal_length   INTEGER                             NOT NULL);
                    """)
        cursor.executescript("""
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
        cursor.executescript("""
          CREATE TABLE IF NOT EXISTS keypoints
             (image_id  INTEGER  PRIMARY KEY  NOT NULL,
              rows      INTEGER               NOT NULL,
              cols      INTEGER               NOT NULL,
              data      BLOB,
          FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
                    """)
        cursor.executescript("""
          CREATE TABLE IF NOT EXISTS descriptors
             (image_id  INTEGER  PRIMARY KEY  NOT NULL,
              rows      INTEGER               NOT NULL,
              cols      INTEGER               NOT NULL,
              data      BLOB,
          FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
                    """)
        cursor.executescript("""
          CREATE TABLE IF NOT EXISTS matches
             (pair_id  INTEGER  PRIMARY KEY  NOT NULL,
              rows     INTEGER               NOT NULL,
              cols     INTEGER               NOT NULL,
              data     BLOB);
                    """)
        cursor.executescript("""
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
        return True

    ##############################
    # insert
    ##############################

    def insert_image(self, image_id, name, camera_id,
                     prior_qw=None, prior_qx=None, prior_qy=None, prior_qz=None,
                     prior_tx=None, prior_ty=None, prior_tz=None):
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO images(image_id, name, camera_id, prior_qw, prior_qx, " +
            "prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) VALUES(?, ?, ?, ?, ?, " +
            "?, ?, ?, ?, ?);", (image_id, name, camera_id, prior_qw, prior_qx,
                                prior_qy, prior_qz, prior_tx, prior_ty, prior_tz))

    def insert_camera(self, camera_id, model, width, height, params, prior_focal_length=1):
        cursor = self.connection.cursor()
        if type(params) == np.ndarray:
            params = params.astype(np.float64).tobytes()

        cursor.execute(
            "INSERT INTO cameras(camera_id, model, width, height, params, " +
            "prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);",
            (camera_id, model, width, height, params, prior_focal_length))

    def insert_keypoints(self, image_id, keypoints):
        if keypoints.shape[1] == 2:
            scale_ori = np.zeros_like(keypoints)
            keypoints = np.concatenate(
                (keypoints, scale_ori), axis=1).astype(np.float32)  # (n, 4)

        cursor = self.connection.cursor()
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) " +
                       "VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1],
                        keypoints.tobytes()))

    def insert_descriptors(self, image_id, descriptors):
        cursor = self.connection.cursor()
        descriptors = descriptors.astype(np.float32)
        cursor.execute("INSERT INTO descriptors(image_id, rows, cols, data) " +
                       "VALUES(?, ?, ?, ?);",
                       (image_id, descriptors.shape[0], descriptors.shape[1],
                        descriptors.tobytes()))

    ##############################
    # query
    ##############################

    def fetch_image(self, image_id):
        """
        image_id, name, camera_id,
        prior_qw, prior_qx, prior_qy, prior_qz,
        prior_tx, prior_ty, prior_tz,
        """
        cursor = self.connection.cursor()
        res = cursor.execute(
            "SELECT * FROM images WHERE image_id={};"
            .format(image_id)).fetchone()
        return res

    def fetch_all_images(self):
        cursor = self.connection.cursor()
        res = cursor.execute(
            "SELECT * FROM images;").fetchall()
        return res

    def fetch_keypoints(self, image_id):
        cursor = self.connection.cursor()
        rows, cols, data = cursor.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id={};"
            .format(image_id)).fetchone()
        kps = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        return kps

    def fetch_descriptors(self, image_id):
        cursor = self.connection.cursor()
        rows, cols, data = cursor.execute(
            "SELECT rows, cols, data FROM descriptors WHERE image_id={};"
            .format(image_id)).fetchone()
        descs = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        return descs

    def fetch_all_keypoints(self):
        """
        Returns
        ---
        a list where each item is in the format (image_id, keypoints)
        """
        cursor = self.connection.cursor()
        res = cursor.execute(
            "SELECT * FROM keypoints;").fetchall()
        kps = []
        for item in res:
            image_id, rows, cols, data = item
            kps.append((image_id, np.frombuffer(data, dtype=np.float32).reshape((rows, cols))[:,:2]))

        return kps

    def fetch_all_descriptors(self):
        """
        Returns
        ---
        a list where each item is in the format (image_id, descriptors)
        """
        cursor = self.connection.cursor()
        res = cursor.execute(
            "SELECT * FROM descriptors;").fetchall()
        descs = []
        for item in res:
            image_id, rows, cols, data = item
            descs.append((image_id, np.frombuffer(data, dtype=np.float32).reshape((rows, cols))))

        return descs

    def close(self):
        self.connection.close()

    def commit(self):
        self.connection.commit()
