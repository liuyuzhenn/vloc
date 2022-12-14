"""
File: model3d.py
Author: liuyuzhen
Email: liuyuzhen22@mails.ucas.ac.cn
Github: https://github.com/liuyuzhenn
Description: 
"""

import time
import numpy as np
import os
import sqlite3


def read_images(file: str):
    images = {}
    with open(file, 'r') as f:
        line = f.readline()
        while line != '':
            if line[0] != '#':
                elements = line.strip('\n').split(' ')
                image_id = int(elements[0])
                pose = np.array([float(e) for e in elements[1:8]])
                camera_id = int(elements[8])
                name = elements[9]

                line = f.readline()
                elements = line.strip('\n').split(' ')
                point3d_ids = np.array([int(elements[i+2])
                                       for i in range(0, len(elements), 3)]).astype(int)
                points2d = np.array([[float(elements[i]), float(elements[i+1])]
                                    for i in range(0, len(elements), 3)]).astype(np.float32)

                image = Image(image_id, pose, camera_id,
                              name, points2d, point3d_ids)

                images[image_id] = image
            line = f.readline()
    return images


def read_points3d(file: str):
    points3d = {}
    with open(file, 'r') as f:
        line = f.readline()
        while line != '':
            if line[0] != '#':
                elements = line.strip('\n').split(' ')
                point3d_id = int(elements[0])
                xyz = np.array([float(e)
                               for e in elements[1:4]]).astype(np.float32)
                rgb = tuple([int(e) for e in elements[4:7]])
                error = float(elements[7])

                track = [tuple([int(elements[i]), int(elements[i+1])])
                         for i in range(8, len(elements), 2)]

                point3d = Point3D(point3d_id, xyz, track, rgb, error)

                points3d[point3d_id] = point3d

            line = f.readline()
    return points3d


def read_cameras(file: str):
    cameras = {}
    with open(file, 'r') as f:
        line = f.readline()
        while line != '':
            if line[0] != '#':
                elements = line.strip('\n').split(' ')
                camera_id = int(elements[0])
                model = elements[1]
                width, height = int(elements[2]), int(elements[3])
                params = np.array([float(e)
                                  for e in elements[4:]]).astype(np.float32)

                camera = Camera(camera_id, model, width, height, params)
                cameras[camera_id] = camera

            line = f.readline()

    return cameras


class Image:
    def __init__(self, image_id, pose, camera_id, name, points2d, point3d_ids):
        self.image_id = image_id
        self.pose = pose
        self.camera_id = camera_id
        self.name = name
        self.points2d = points2d
        self.point3d_ids = point3d_ids


class Camera:
    def __init__(self, camera_id: int, model: str,
                 width: int, height: int, params: np.ndarray):
        self.camera_id = camera_id
        self.model = model
        self.width = width
        self.height = height
        self.params = params


class Point3D:
    def __init__(self, point_id: int, point3d: np.ndarray, track: list = [],
                 color: tuple = (0, 0, 0), error: float = -1.0, descriptor=None):
        self.point_id = point_id
        self.point3d = point3d
        self.track = track  # each element is a tuple: (image_id, point2d_id)
        self.color = color
        self.error = error
        self.descriptor = descriptor


class Model3D:
    """Store 3D points and their corresponding tracks. """

    def __init__(self, folder: str, database: str, desc_mode='mean'):
        self.folder = folder
        self.database = database
        self.images = read_images(os.path.join(folder, 'images.txt'))
        self.cameras = read_cameras(os.path.join(folder, 'cameras.txt'))
        self.points3d = read_points3d(os.path.join(folder, 'points3D.txt'))
        print('Model information: ')
        print('  ==> images : {}'.format(len(self.images)))
        print('  ==> points : {}'.format(len(self.points3d)))
        print('  ==> cameras: {}'.format(len(self.cameras)))

        # load descriptors
        connection = sqlite3.connect(database)
        cursor = connection.cursor()
        dic_imgid_desc = cursor.execute(
            "SELECT image_id,rows,cols,data FROM descriptors;").fetchall()
        connection.close()

        # image_id, descriptor
        dic_imgid_desc = {e[0]: np.frombuffer(e[3], dtype=np.float32).reshape(
            (e[1], e[2])) for e in dic_imgid_desc}
        for point3d in self.points3d.values():
            track = point3d.track
            # indices in colmap start from 0
            descs = np.stack([dic_imgid_desc[image_id][point2d_id] for (image_id, point2d_id)
                              in track], axis=0)
            if desc_mode == 'mean':
                desc_mean = np.mean(descs, axis=0)
            elif desc_mode == 'median':
                desc_mean = np.median(descs, axis=0)
            else:
                raise NotImplementedError
            desc_mean_manifold = desc_mean/np.linalg.norm(desc_mean)
            point3d.descriptor = desc_mean_manifold


def test():
    name = 'Fort_Channing_gate'
    image_file = '../../data/{}/sparse/0/images.txt'.format(name)
    point_file = '../../data/{}/sparse/0/points3D.txt'.format(name)
    camera_file = '../../data/{}/sparse/0/cameras.txt'.format(name)
    t1 = time.time()
    images = read_images(image_file)
    t2 = time.time()
    points = read_points3d(point_file)
    t3 = time.time()
    cameras = read_cameras(camera_file)

    print('Loading {} images costs {:.3f} s'.format(len(images), t2-t1))
    print('Loading {} 3D points costs {:.3f} s'.format(len(points), t3-t2))
    image = images[1]
    s = 0


def test_model3d():
    name = 'Buddah_tooth_relic_temple'
    database = '../../data/{}/database.db'.format(name)
    folder = '../../data/{}/sparse/0/'.format(name)
    t1 = time.time()
    model3d = Model3D(folder, database)
    t2 = time.time()
    print('Loading model takes {:.1f}s'.format(t2-t1))


if __name__ == '__main__':
    test_model3d()
    s = 1
