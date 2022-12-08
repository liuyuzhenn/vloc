import sqlite3
import os
import shutil
import numpy as np


def get_all_items(cur, s):
    res = cur.execute("SELECT * FROM {}".format(s)).fetchall()
    return res


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


root_dir = 'C:/Users/23792/Desktop'
src_db_file = os.path.join(root_dir, 'test.db')
db_file = os.path.join('.', 'test.db')
# os.remove(db_file)
shutil.copyfile(src_db_file, db_file)

con = sqlite3.connect(db_file)
cur = con.cursor()
res = cur.execute("SELECT rows,cols FROM matches;").fetchone()
print(res)

m = 0
