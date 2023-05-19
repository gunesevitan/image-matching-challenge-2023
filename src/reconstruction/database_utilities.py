import os
import h5py
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import sqlite3

import camera_utilities


MAX_IMAGE_ID = 2 ** 31 - 1

CREATE_CAMERAS_TABLE_QUERY = '''
CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL
)
'''

CREATE_IMAGES_TABLE_QUERY = f'''
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {MAX_IMAGE_ID}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
)
'''

CREATE_DESCRIPTORS_TABLE_QUERY = '''
CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
)
'''

CREATE_KEYPOINTS_TABLE_QUERY = '''CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
)
'''

CREATE_MATCHES_TABLE_QUERY = '''
CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB
)
'''

CREATE_TWO_VIEW_GEOMETRIES_TABLE_QUERY = '''
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB
)
'''

CREATE_NAME_INDEX_QUERY = 'CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)'

CREATE_ALL_QUERY = '; '.join([
    CREATE_CAMERAS_TABLE_QUERY,
    CREATE_IMAGES_TABLE_QUERY,
    CREATE_KEYPOINTS_TABLE_QUERY,
    CREATE_DESCRIPTORS_TABLE_QUERY,
    CREATE_MATCHES_TABLE_QUERY,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE_QUERY,
    CREATE_NAME_INDEX_QUERY
])


def blob_to_array(x):
    if x is not None:
        return np.frombuffer(x)


def array_to_blob(x):
    return np.asarray(x, np.float64).tobytes()


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):

        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL_QUERY)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE_QUERY)
        self.create_descriptors_table = lambda: self.executescript(CREATE_DESCRIPTORS_TABLE_QUERY)
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE_QUERY)
        self.create_two_view_geometries_table = lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE_QUERY)
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE_QUERY)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE_QUERY)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX_QUERY)

    def get_cameras_table(self):

        df_cameras = pd.read_sql('SELECT * FROM cameras', con=self)
        if df_cameras.shape[0] > 0:
            df_cameras['params'] = df_cameras['params'].apply(lambda x: blob_to_array(x))

        return df_cameras

    def get_images_table(self):

        df_images = pd.read_sql('SELECT * FROM images', con=self)

        return df_images

    def get_descriptors_table(self):

        df_descriptors = pd.read_sql('SELECT * FROM descriptors', con=self)
        if df_descriptors.shape[0] > 0:
            df_descriptors['data'] = df_descriptors['data'].apply(lambda x: blob_to_array(x))

        return df_descriptors

    def get_keypoints_table(self):

        df_keypoints = pd.read_sql('SELECT * FROM keypoints', con=self)
        if df_keypoints.shape[0] > 0:
            df_keypoints['data'] = df_keypoints['data'].apply(lambda x: blob_to_array(x))

        return df_keypoints

    def get_matches_table(self):

        df_matches = pd.read_sql('SELECT * FROM matches', con=self)
        if df_matches.shape[0] > 0:
            df_matches['data'] = df_matches['data'].apply(lambda x: blob_to_array(x))

        return df_matches

    def get_two_view_geometries_table(self):

        df_two_view_geometries = pd.read_sql('SELECT * FROM two_view_geometries', con=self)
        if df_two_view_geometries.shape[0] > 0:
            df_two_view_geometries['data'] = df_two_view_geometries['data'].apply(lambda x: blob_to_array(x))
            df_two_view_geometries['F'] = df_two_view_geometries['F'].apply(lambda x: blob_to_array(x))
            df_two_view_geometries['E'] = df_two_view_geometries['E'].apply(lambda x: blob_to_array(x))
            df_two_view_geometries['H'] = df_two_view_geometries['H'].apply(lambda x: blob_to_array(x))
            df_two_view_geometries['qvec'] = df_two_view_geometries['qvec'].apply(lambda x: blob_to_array(x))
            df_two_view_geometries['tvec'] = df_two_view_geometries['tvec'].apply(lambda x: blob_to_array(x))

        return df_two_view_geometries

    def add_camera(self, camera_id, model, width, height, params, prior_focal_length):

        cursor = self.execute(
            'INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)',
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length
            )
        )

        return cursor.lastrowid

    def add_image(self, image_id, name, camera_id, prior_q=np.zeros(4), prior_t=np.zeros(3), ):

        cursor = self.execute(
            'INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                image_id,
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2]
            )
        )

        return cursor.lastrowid

    def add_descriptors(self, image_id, descriptors):

        descriptors = np.ascontiguousarray(descriptors, np.uint8)

        self.execute(
            'INSERT INTO descriptors VALUES (?, ?, ?, ?)',
            (
                    image_id,
                    descriptors.shape[0],
                    descriptors.shape[1],
                    array_to_blob(descriptors)
            )
        )

    def add_keypoints(self, image_id, keypoints):

        keypoints = np.asarray(keypoints, np.float32)

        self.execute(
            'INSERT INTO keypoints VALUES (?, ?, ?, ?)',
            (
                image_id,
                keypoints.shape[0],
                keypoints.shape[1],
                array_to_blob(keypoints)
            )
        )

    def add_matches(self, image1_id, image2_id, matches):

        if image1_id > image2_id:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image1_id, image2_id)
        matches = np.asarray(matches, np.uint32)

        self.execute(
            'INSERT INTO matches VALUES (?, ?, ?, ?)',
            (
                pair_id,
                matches.shape[0],
                matches.shape[1],
                array_to_blob(matches)
            )
        )

    def add_two_view_geometries(self, image1_id, image2_id, matches, config=2, F=np.eye(3), E=np.eye(3), H=np.eye(3), ):

        if image1_id > image2_id:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image1_id, image2_id)
        matches = np.asarray(matches, np.uint32)

        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)

        self.execute(
            'INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                pair_id,
                matches.shape[0],
                matches.shape[1],
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H)
            )
        )


def create_camera(colmap_database, image_path, camera_model):

    image = Image.open(image_path)
    width, height = image.size

    focal_length = camera_utilities.get_focal_length(image_path=image_path)

    if camera_model == 'simple-pinhole':
        model = 0
        params = np.array([focal_length, width / 2, height / 2])
    elif camera_model == 'pinhole':
        model = 1
        params = np.array([focal_length, focal_length, width / 2, height / 2])
    elif camera_model == 'simple-radial':
        model = 2
        params = np.array([focal_length, width / 2, height / 2, 0.1])
    elif camera_model == 'opencv':
        model = 4
        params = np.array([focal_length, focal_length, width / 2, height / 2, 0., 0., 0., 0.])
    else:
        raise ValueError(f'Invalid camera model: {camera_model}')

    return colmap_database.add_camera(
        camera_id=None,
        model=model,
        width=width,
        height=height,
        params=params,
        prior_focal_length=False
    )


def add_keypoints(colmap_database, h5_path, image_path, img_ext, camera_model, single_camera = True):

    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]

        fname_with_ext = filename# + img_ext
        path = os.path.join(image_path, fname_with_ext)
        if not os.path.isfile(path):
            raise IOError(f'Invalid image path {path}')

        if camera_id is None or not single_camera:
            camera_id = create_camera(colmap_database, path, camera_model)
        image_id = colmap_database.add_image(fname_with_ext, camera_id)
        fname_to_id[filename] = image_id

        colmap_database.add_keypoints(image_id, keypoints)

    return fname_to_id
