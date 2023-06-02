import h5py
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import torch
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
    return np.asarray(x).tobytes()


def image_ids_to_pair_id(image1_id, image2_id):
    if image1_id > image2_id:
        image1_id, image2_id = image2_id, image1_id
    return image1_id * MAX_IMAGE_ID + image2_id


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path, uri):
        return sqlite3.connect(database_path, uri=uri, factory=COLMAPDatabase)

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

    def add_image(self, image_id, name, camera_id, prior_q=np.zeros(4), prior_t=np.zeros(3)):

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

    def add_two_view_geometries(self, image1_id, image2_id, matches, config=2, F=np.eye(3), E=np.eye(3), H=np.eye(3)):

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


def get_first_indices(x, dim=0):

    """
    Get indices where values appear first

    Parameters
    ----------
    x: torch.Tensor
        N-dimensional torch tensor

    dim: int
        Dimension of the unique operation

    Returns
    -------
    first_indices: torch.Tensor
        Indices where values appear first
    """

    _, idx, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, sorted_idx = torch.sort(idx, stable=True)
    counts_cumsum = counts.cumsum(0)
    counts_cumsum = torch.cat((torch.tensor([0], device=counts_cumsum.device), counts_cumsum[:-1]))
    first_indices = sorted_idx[counts_cumsum]

    return first_indices


def write_matches(image_paths, image_pair_indices, first_image_keypoints, second_image_keypoints, output_directory):

    """
    Write matches as h5 datasets for COLMAP

    Parameters
    ----------
    image_paths: list of shape (n_images)
        List of image paths

    image_pair_indices: list of shape (n_image_pairs)
        List of tuples of image pair indices

    first_image_keypoints: list of shape (n_image_pairs)
        List of first image keypoints

    second_image_keypoints: list of shape (n_image_pairs)
        List of second image keypoints

    output_directory: str or pathlib.Path object
        Path of the output directory
    """

    with h5py.File(output_directory / 'model_matches.h5', mode='w') as f:
        for matching_index, image_pair_index in enumerate(image_pair_indices):
            first_image_path, second_image_path = image_paths[image_pair_index[0]], image_paths[image_pair_index[1]]
            first_image_filename, second_image_filename = first_image_path.split('/')[-1], second_image_path.split('/')[-1]

            # Concatenate matched keypoints of an image pair and write it as a dataset
            group = f.require_group(first_image_filename)
            group.create_dataset(
                second_image_filename,
                data=np.concatenate([first_image_keypoints[matching_index], second_image_keypoints[matching_index]], axis=1)
            )

    keypoints = defaultdict(list)
    match_indices = defaultdict(dict)
    total_keypoints = defaultdict(int)

    with h5py.File(output_directory / 'model_matches.h5', mode='r') as f:
        for first_image_filename in f.keys():
            group = f[first_image_filename]
            for second_image_filename in group.keys():
                image_pair_keypoints = group[second_image_filename][...]
                keypoints[first_image_filename].append(image_pair_keypoints[:, :2])
                keypoints[second_image_filename].append(image_pair_keypoints[:, 2:])
                current_match = torch.arange(len(image_pair_keypoints)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_keypoints[first_image_filename]
                current_match[:, 1] += total_keypoints[second_image_filename]
                total_keypoints[first_image_filename] += len(image_pair_keypoints)
                total_keypoints[second_image_filename] += len(image_pair_keypoints)
                match_indices[first_image_filename][second_image_filename] = current_match

    for image_filename in keypoints.keys():
        keypoints[image_filename] = np.round(np.concatenate(keypoints[image_filename], axis=0))

    unique_keypoints = {}
    unique_match_indices = {}
    matches = defaultdict(dict)

    for image_filename in keypoints.keys():
        unique_keypoint_values, unique_keypoint_reverse_idx = torch.unique(torch.from_numpy(keypoints[image_filename]), dim=0, return_inverse=True)
        unique_match_indices[image_filename] = unique_keypoint_reverse_idx
        unique_keypoints[image_filename] = unique_keypoint_values.numpy()

    for first_image_filename, group in match_indices.items():
        for second_image_filename, image_pair_match_index in group.items():
            image_pair_match_index_copy = deepcopy(image_pair_match_index)
            image_pair_match_index_copy[:, 0] = unique_match_indices[first_image_filename][image_pair_match_index_copy[:, 0]]
            image_pair_match_index_copy[:, 1] = unique_match_indices[second_image_filename][image_pair_match_index_copy[:, 1]]
            matched_keypoints = np.concatenate([
                unique_keypoints[first_image_filename][image_pair_match_index_copy[:, 0]].reshape(-1, 2),
                unique_keypoints[second_image_filename][image_pair_match_index_copy[:, 1]].reshape(-1, 2)
            ], axis=1)

            if matched_keypoints.shape[0] == 0:
                continue

            current_unique_match_index = get_first_indices(torch.from_numpy(matched_keypoints), dim=0)
            image_pair_match_index_copy_semiclean = image_pair_match_index_copy[current_unique_match_index]

            current_unique_match_index1 = get_first_indices(image_pair_match_index_copy_semiclean[:, 0], dim=0)
            image_pair_match_index_copy_semiclean = image_pair_match_index_copy_semiclean[current_unique_match_index1]

            current_unique_match_index2 = get_first_indices(image_pair_match_index_copy_semiclean[:, 1], dim=0)
            image_pair_match_index_copy_semiclean2 = image_pair_match_index_copy_semiclean[current_unique_match_index2]

            matches[first_image_filename][second_image_filename] = image_pair_match_index_copy_semiclean2.numpy()

    with h5py.File(output_directory / 'keypoints.h5', mode='w') as f:
        for image_filename, keypoints in unique_keypoints.items():
            f[image_filename] = keypoints

    with h5py.File(output_directory / 'matches.h5', mode='w') as f:
        for first_image_filename, first_image_matches in matches.items():
            group = f.require_group(first_image_filename)
            for second_image_filename, second_image_matches in first_image_matches.items():
                group[second_image_filename] = second_image_matches


def push_to_database(colmap_database, dataset_directory, image_directory, camera_model, single_camera):

    """
    Push cameras, images, keypoints and matches to COLMAP database

    Parameters
    ----------
    colmap_database: COLMAPDatabase
        COLMAP Database object

    dataset_directory: str or pathlib.Path
        Reconstruction directory

    image_directory: str or pathlib.Path
        Image directory

    camera_model: str
        Model of the camera

    single_camera: bool
        Whether there is one or multiple in cameras
    """

    keypoints_dataset = h5py.File(dataset_directory / 'keypoints.h5', 'r')

    camera_id = None
    image_filename_to_id = {}

    for image_filename in tqdm(list(keypoints_dataset.keys())):

        keypoints = keypoints_dataset[image_filename][()]

        if camera_id is None or not single_camera:

            image = Image.open(str(image_directory / image_filename))
            width, height = image.size

            focal_length = camera_utilities.get_focal_length(image_path=str(image_directory / image_filename))

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

            camera_id = colmap_database.add_camera(
                camera_id=None,
                model=model,
                width=width,
                height=height,
                params=params,
                prior_focal_length=False
            )

        image_id = colmap_database.add_image(image_id=None, name=image_filename, camera_id=camera_id)
        image_filename_to_id[image_filename] = image_id

        colmap_database.add_keypoints(image_id=image_id, keypoints=keypoints)

    matches_dataset = h5py.File(dataset_directory / 'matches.h5', 'r')

    pairs = set()

    for image1_filename in matches_dataset.keys():
        group = matches_dataset[image1_filename]
        for image2_filename in group.keys():
            image1_id = image_filename_to_id[image1_filename]
            image2_id = image_filename_to_id[image2_filename]
            pair_id = image_ids_to_pair_id(image1_id=image1_id, image2_id=image2_id)
            if pair_id in pairs:
                continue

            matches = group[image2_filename][()]
            colmap_database.add_matches(image1_id, image2_id, matches)
            pairs.add(pair_id)

    colmap_database.commit()
