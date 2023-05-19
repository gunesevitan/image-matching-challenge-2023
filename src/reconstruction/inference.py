import sys
import os
import shutil
import argparse
import yaml
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from kornia.feature import LoFTR
import pycolmap
import sqlite3

sys.path.append('..')
import settings
import evaluation
import visualization
import datasets
import image_selection
import image_matching


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.model_directory / 'config.yaml', 'r'), Loader=yaml.FullLoader)

    # Load image selection model with specified configurations
    image_selection_device = torch.device(config['image_selection']['device'])
    image_selection_model = image_selection.load_feature_extractor(**config['image_selection']['model'])
    image_selection_model.to(image_selection_device)
    image_selection_model.eval()
    image_selection_transforms = image_selection.create_image_selection_transforms(**config['image_selection']['transforms'])

    # Load image matching model with specified configurations
    image_matching_device = torch.device(config['image_matching']['device'])
    image_matching_model = LoFTR(config['image_matching']['pretrained'])
    image_matching_model.to(image_matching_device)
    image_matching_model.eval()
    image_matching_transforms = config['image_matching']['transforms']

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

                scene_reconstruction_directory = reconstruction_root_directory / dataset / scene
                scene_reconstruction_directory.mkdir(parents=True, exist_ok=True)
                database_path = scene_reconstruction_directory / 'database.db'

                for file_or_directory in os.listdir(scene_reconstruction_directory):
                    # Remove files and directories from the previous reconstruction
                    file_or_directory_path = scene_reconstruction_directory / file_or_directory
                    if file_or_directory_path.is_file():
                        os.remove(file_or_directory_path)
                    elif file_or_directory_path.is_dir():
                        shutil.rmtree(file_or_directory_path)

                scene_image_count = len(image_paths)
                settings.logger.info(
                    f'''
                    Dataset: {dataset} - Scene: {scene}
                    Image count: {scene_image_count}
                    '''
                )

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
                    image_paths = image_selection.select_images(
                        image_paths=image_paths,
                        image_selection_features=image_selection_features,
                        image_count=config['image_selection']['image_count']
                    )

                    del image_selection_device, image_selection_model, image_selection_transforms, image_selection_data_loader, image_selection_features
                    settings.logger.info(f'Selected most similar {len(image_paths)} images')

                image_pair_indices = image_matching.create_image_pairs(image_paths=image_paths)
                image_matching_data_loader = image_matching.prepare_dataloader(
                    image_paths=image_paths,
                    image_pair_indices=image_pair_indices,
                    transforms=image_matching_transforms,
                    batch_size=config['image_matching']['batch_size'],
                    num_workers=config['image_matching']['num_workers']
                )
                first_image_keypoints = []
                second_image_keypoints = []
                confidences = []

                for idx, (first_images, second_images) in enumerate(tqdm(image_matching_data_loader)):

                    first_images = first_images.to(image_matching_device)
                    second_images = second_images.to(image_matching_device)
                    outputs = image_matching.match_images(
                        first_images=first_images,
                        second_images=second_images,
                        model=image_matching_model,
                        device=image_matching_device,
                        amp=False
                    )

                    for batch_index in np.unique(outputs['batch_indexes']):
                        batch_mask = outputs['batch_indexes'] == batch_index
                        first_image_keypoints.append(outputs['keypoints0'][batch_mask])
                        second_image_keypoints.append(outputs['keypoints1'][batch_mask])
                        confidences.append(outputs['confidence'][batch_mask])

                del image_matching_device, image_matching_model, image_matching_transforms, image_matching_data_loader
                settings.logger.info(f'Finished matching {len(image_pair_indices)} image pairs')

                
                exit()
                # TODO: Push keypoints to COLMAP database
                # Find max image size from selected image sizes
                max_image_size = df.loc[(df['scene'] == scene) & (df['image_id'].isin(image_list)), ['image_height', 'image_width']].values.min()
                max_image_size = 1400

                sift_extraction_options = pycolmap.SiftExtractionOptions(**config['sift_extraction'])
                sift_extraction_options.max_image_size = int(max_image_size)

                pycolmap.extract_features(
                    database_path=database_path,
                    image_path=scene_directory / 'images',
                    image_list=image_list,
                    sift_options=sift_extraction_options,
                    device=pycolmap.Device(config['colmap']['device']),
                    verbose=True
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

                con = sqlite3.connect(database_path)
                cursor = con.cursor()
                # [('cameras',), ('sqlite_sequence',), ('images',), ('keypoints',), ('descriptors',), ('matches',), ('two_view_geometries',)]
                df_db = pd.read_sql_query("SELECT * FROM keypoints", con)

                break

        df = df.dropna(subset=['rotation_matrix_prediction', 'translation_vector_prediction'])

        df_scene_scores = evaluation.evaluate(df=df, verbose=True)
        df_dataset_scores = df_scene_scores.groupby('dataset')[['maa', 'rotation_maa', 'translation_maa']].mean().reset_index()
        df_scores = df_dataset_scores[['maa', 'rotation_maa', 'translation_maa']].mean()

        df_scene_scores.to_csv(reconstruction_root_directory / 'scene_scores.csv', index=False)
        df_dataset_scores.to_csv(reconstruction_root_directory / 'dataset_scores.csv', index=False)
        df_scores.to_csv(reconstruction_root_directory / 'scores.csv', index=True, header=None)
        settings.logger.info(f'scene_scores.csv, dataset_scores.csv and scores.csv are saved to {reconstruction_root_directory}')
