import sys
import os
import shutil
import argparse
import yaml
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
from kornia.feature import LoFTR
import pycolmap

sys.path.append('..')
import settings
import evaluation
import image_utilities
import image_selection
import loftr
import superglue
import database_utilities
sys.path.append(str(settings.ROOT / 'venv' / 'lib' / 'python3.9' / 'site-packages' / 'SuperGluePretrainedNetwork'))
from models.matching import Matching


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.model_directory / 'config.yaml', 'r'), Loader=yaml.FullLoader)

    # Load image selection model with specified configurations
    image_selection_device = torch.device(config['image_selection']['device'])
    image_selection_model = image_selection.load_feature_extractor(**config['image_selection']['model'])
    image_selection_model = image_selection_model.eval().to(image_selection_device)
    image_selection_transforms = image_selection.create_image_selection_transforms(**config['image_selection']['transforms'])

    # Load LoFTR model with specified configurations
    image_matching_device = torch.device(config['image_matching']['device'])
    loftr_model = LoFTR(config['loftr']['pretrained'])
    loftr_model.load_state_dict(torch.load(config['loftr']['pretrained_weights_path'])['state_dict'])
    loftr_model = loftr_model.eval().to(image_matching_device)
    loftr_transforms = config['loftr']['transforms']

    # Load SuperPoint and SuperGlue model with specified configurations
    superglue_model = Matching({'superglue': config['superglue'], 'superpoint': config['superpoint']})
    superglue_model = superglue_model.eval().to(image_matching_device)
    superglue_transforms = config['superglue']['transforms']

    reconstruction_root_directory = settings.MODELS / config['persistence']['root_directory']
    reconstruction_root_directory.mkdir(parents=True, exist_ok=True)

    if args.mode == 'validation':

        settings.logger.info('Running reconstruction on validation mode')

        df = pd.read_csv(settings.DATA / 'train' / 'train_metadata.csv')
        df['image_id'] = df['image_path'].apply(lambda x: str(x).split('/')[-1])

        settings.logger.info(
            f'''
            Train Labels shape: {df.shape}
            Unique dataset count: {df['dataset'].nunique()}
            Unique scene count: {df['scene'].nunique()}
            Mean image count per scene: {df.groupby('scene')['scene'].count().mean():.4f} 
            '''
        )

        for dataset in config['dataset']:

            dataset_directory = settings.DATA / 'train' / dataset

            for scene in config['dataset'][dataset]:

                scene_directory = dataset_directory / scene
                image_paths = sorted(glob(str(scene_directory / config['dataset'][dataset][scene]['image_directory'] / '*')))
                scene_image_count = len(image_paths)
                settings.logger.info(
                    f'''
                    Dataset: {dataset} - Scene: {scene}
                    Image count: {scene_image_count}
                    '''
                )

                scene_reconstruction_directory = reconstruction_root_directory / dataset / scene
                scene_reconstruction_directory.mkdir(parents=True, exist_ok=True)

                for file_or_directory in os.listdir(scene_reconstruction_directory):
                    # Remove files and directories from the previous reconstruction
                    file_or_directory_path = scene_reconstruction_directory / file_or_directory
                    if file_or_directory_path.is_file():
                        os.remove(file_or_directory_path)
                    elif file_or_directory_path.is_dir():
                        shutil.rmtree(file_or_directory_path)

                # Create COLMAP database and its tables for the current reconstruction
                database_path = scene_reconstruction_directory / 'database.db'
                database_uri = f'file:{database_path}?mode=rwc'
                colmap_database = database_utilities.COLMAPDatabase.connect(database_uri, uri=True)
                colmap_database.create_tables()

                # Select images if scene image count is above the specified threshold
                if scene_image_count > config['image_selection']['image_count']:

                    image_selection_data_loader = image_selection.prepare_dataloader(
                        image_paths=image_paths,
                        transforms=image_selection_transforms,
                        batch_size=config['image_selection']['batch_size'],
                        num_workers=config['image_selection']['num_workers']
                    )
                    image_selection_features = []

                    for idx, inputs in enumerate(tqdm(image_selection_data_loader)):

                        inputs = inputs.to(image_selection_device)
                        batch_image_selection_features = image_selection.extract_features(
                            inputs=inputs,
                            model=image_selection_model,
                            pooling_type=config['image_selection']['pooling_type'],
                            device=image_selection_device,
                            amp=config['image_selection']['amp']
                        )
                        image_selection_features.append(batch_image_selection_features)

                    image_selection_features = torch.cat(image_selection_features, dim=0).numpy()

                    # Select images with the highest mean cosine similarity because they are more likely to be registered
                    image_paths = image_selection.select_images(
                        image_paths=image_paths,
                        image_selection_features=image_selection_features,
                        image_count=config['image_selection']['image_count']
                    )
                    settings.logger.info(f'Selected most similar {len(image_paths)} images')

                    del image_selection_device, image_selection_model, image_selection_transforms, image_selection_data_loader, image_selection_features

                # Create brute force image pairs from image paths
                image_pair_indices = image_utilities.create_image_pairs(image_paths=image_paths)
                first_image_keypoints = []
                second_image_keypoints = []

                for image_pair_idx, (first_image_idx, second_image_idx) in enumerate(tqdm(image_pair_indices)):

                    image1 = cv2.imread(str(image_paths[image_pair_indices[image_pair_idx][0]]))
                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

                    image2 = cv2.imread(str(image_paths[image_pair_indices[image_pair_idx][1]]))
                    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

                    loftr_outputs = loftr.match_images(
                        image1=image1,
                        image2=image2,
                        model=loftr_model,
                        device=image_matching_device,
                        amp=False,
                        transforms=loftr_transforms
                    )

                    superglue_outputs = superglue.match_images(
                        image1=image1,
                        image2=image2,
                        model=superglue_model,
                        device=image_matching_device,
                        amp=False,
                        transforms=superglue_transforms
                    )

                    image1_keypoints = np.concatenate([
                        loftr_outputs['keypoints0'],
                        superglue_outputs['keypoints0']
                    ])

                    image2_keypoints = np.concatenate([
                        loftr_outputs['keypoints1'],
                        superglue_outputs['keypoints1']
                    ])

                    first_image_keypoints.append(image1_keypoints)
                    second_image_keypoints.append(image2_keypoints)

                database_utilities.write_matches(
                    image_paths=image_paths,
                    image_pair_indices=image_pair_indices,
                    first_image_keypoints=first_image_keypoints,
                    second_image_keypoints=second_image_keypoints,
                    output_directory=scene_reconstruction_directory
                )

                database_utilities.push_to_database(
                    colmap_database=colmap_database,
                    dataset_directory=scene_reconstruction_directory,
                    image_directory=scene_directory / 'images',
                    camera_model='simple-radial',
                    single_camera=True
                )

                sift_matching_options = pycolmap.SiftMatchingOptions(**config['sift_matching'])
                exhaustive_matching_options = pycolmap.ExhaustiveMatchingOptions(**config['exhaustive_matching'])

                pycolmap.match_exhaustive(
                    database_path=database_path,
                    sift_options=sift_matching_options,
                    matching_options=exhaustive_matching_options,
                    device=pycolmap.Device(config['colmap']['device']),
                    verbose=True
                )

                incremental_mapper_options = pycolmap.IncrementalMapperOptions(**config['incremental_mapper'])

                reconstructions = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=scene_directory / 'images',
                    output_path=scene_reconstruction_directory,
                    options=incremental_mapper_options
                )

                if len(reconstructions) > 0:

                    best_registered_image_count = 0
                    best_reconstruction_idx = None

                    for reconstruction_idx in reconstructions.keys():
                        if reconstructions[reconstruction_idx].num_reg_images() > best_registered_image_count:
                            best_reconstruction_idx = reconstruction_idx
                            best_registered_image_count = reconstructions[reconstruction_idx].num_reg_images()

                    best_reconstruction = reconstructions[best_reconstruction_idx]
                else:
                    best_registered_image_count = 0
                    best_reconstruction_idx = None
                    best_reconstruction = None

                settings.logger.info(
                    f'''
                    Dataset: {dataset} - Scene: {scene}
                    Reconstruction count: {len(reconstructions)}
                    Best reconstruction registered image count: {best_registered_image_count}
                    '''
                )

                if best_reconstruction is not None:
                    registered_images = {image.name: image for image in best_reconstruction.images.values()}
                else:
                    registered_images = {}

                for idx, row in df.loc[df['scene'] == scene].iterrows():
                    if row['image_id'] in registered_images:
                        rotation_matrix_prediction = registered_images[row['image_id']].rotmat()
                        translation_vector_prediction = registered_images[row['image_id']].tvec
                    else:
                        rotation_matrix_prediction = np.zeros((3, 3))
                        translation_vector_prediction = np.zeros((3, 1))

                    df.loc[idx, 'rotation_matrix_prediction'] = ';'.join([str(x) for x in rotation_matrix_prediction.reshape(-1)])
                    df.loc[idx, 'translation_vector_prediction'] = ';'.join([str(x) for x in translation_vector_prediction.reshape(-1)])

        df = df.dropna(subset=['rotation_matrix_prediction', 'translation_vector_prediction'])

        df_scene_scores = evaluation.evaluate(df=df, verbose=True)
        df_dataset_scores = df_scene_scores.groupby('dataset')[['maa', 'rotation_maa', 'translation_maa']].mean().reset_index()
        df_scores = df_dataset_scores[['maa', 'rotation_maa', 'translation_maa']].mean()

        df_scene_scores.to_csv(reconstruction_root_directory / 'scene_scores.csv', index=False)
        df_dataset_scores.to_csv(reconstruction_root_directory / 'dataset_scores.csv', index=False)
        df_scores.to_csv(reconstruction_root_directory / 'scores.csv', index=True, header=None)
        settings.logger.info(f'scene_scores.csv, dataset_scores.csv and scores.csv are saved to {reconstruction_root_directory}')
