import sys
import os
import argparse
import yaml
import pathlib
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics.pairwise import cosine_similarity
from kornia.feature import LoFTR

sys.path.append('..')
import settings
import visualization
import image_utilities
import pair_utilities
import datasets
import transforms
import feature_extraction
import loftr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.model_directory / 'config.yaml', 'r'), Loader=yaml.FullLoader)

    if args.mode == 'validation':

        settings.logger.info('Running inference on validation mode')

        for dataset in config['dataset']:

            dataset_directory = settings.DATA / 'train' / dataset

            for scene in config['dataset'][dataset]:

                scene_directory = dataset_directory / scene

                if 'images_full' in os.listdir(str(scene_directory)):
                    image_paths = sorted(glob(str(scene_directory / 'images' / '*')))
                else:
                    image_paths = sorted(glob(str(scene_directory / 'images' / '*')))

                scene_image_count = len(image_paths)
                settings.logger.info(
                    f'''
                    Dataset: {dataset} - Scene: {scene}
                    Image Count: {scene_image_count}
                    '''
                )

                if scene_image_count > config['pair']['image_count_threshold']:

                    # Create image pairs from a sampled subset based on image similarity in the scene
                    feature_extraction_transforms = transforms.create_feature_extraction_transforms(**config['transforms']['feature_extraction'])
                    feature_extraction_dataset = datasets.ImageDataset(image_paths=image_paths, transforms=feature_extraction_transforms)
                    feature_extraction_data_loader = DataLoader(
                        feature_extraction_dataset,
                        batch_size=config['inference']['feature_extraction']['batch_size'],
                        sampler=SequentialSampler(feature_extraction_dataset),
                        pin_memory=False,
                        drop_last=False,
                        num_workers=config['inference']['feature_extraction']['num_workers']
                    )
                    feature_extraction_device = torch.device(config['inference']['feature_extraction']['device'])
                    feature_extraction_model = feature_extraction.load_timm_model(
                        model_name=config['model']['feature_extraction']['model_name'],
                        pretrained=config['model']['feature_extraction']['pretrained'],
                        model_args=config['model']['feature_extraction']['model_args']
                    )

                    features = []

                    for inputs in tqdm(feature_extraction_data_loader):

                        batch_features = feature_extraction.extract_features(
                            inputs=inputs,
                            model=feature_extraction_model,
                            pooling_type=config['inference']['feature_extraction']['pooling_type'],
                            device=feature_extraction_device,
                            amp=config['inference']['feature_extraction']['amp']
                        )
                        features.append(batch_features)

                    del feature_extraction_transforms, feature_extraction_dataset, feature_extraction_data_loader
                    del feature_extraction_device, feature_extraction_model

                    features = torch.cat(features, dim=0).numpy()
                    # Calculate pairwise cosine similarities between features
                    pairwise_cosine_similarities = cosine_similarity(features)
                    # Zero the diagonal and select the upper triangle of cosine similarities for eliminating duplicate pairs
                    np.fill_diagonal(pairwise_cosine_similarities, 0)
                    pairwise_cosine_similarities = np.triu(pairwise_cosine_similarities)

                    # Take image pairs with greater than specified similarity threshold
                    image_similarity_threshold = config['pair']['image_similarity_threshold']
                    image_pair_indices = np.array(np.where(pairwise_cosine_similarities >= image_similarity_threshold)).T.tolist()
                    settings.logger.info(f'{len(image_pair_indices)} pairs (>= {image_similarity_threshold} similarity threshold) are created from {scene_image_count} images')
                    del features, pairwise_cosine_similarities

                else:
                    # Create image pairs from all images in the scene
                    image_pair_indices = pair_utilities.create_all_image_pairs(image_paths)
                    settings.logger.info(f'{len(image_pair_indices)} pairs are created from {scene_image_count} images')

                image_matching_device = torch.device(config['inference']['image_matching']['device'])
                loftr_model = LoFTR(config['model']['image_matching']['pretrained'])
                loftr_model.to(image_matching_device)
                loftr_model.eval()

                for image_1_idx, image_2_idx in tqdm(image_pair_indices):

                    # Create image tensors from the image pair and match them using LoFTR model
                    image1 = image_utilities.get_image_tensor(
                        image_path_or_array=image_paths[image_1_idx],
                        resize_shape=config['transforms']['image_matching']['resize_shape'],
                        resize_longest_edge=config['transforms']['image_matching']['resize_longest_edge'],
                        grayscale=config['transforms']['image_matching']['grayscale'],
                        device=image_matching_device
                    )
                    image2 = image_utilities.get_image_tensor(
                        image_path_or_array=image_paths[image_2_idx],
                        resize_shape=config['transforms']['image_matching']['resize_shape'],
                        resize_longest_edge=config['transforms']['image_matching']['resize_longest_edge'],
                        grayscale=config['transforms']['image_matching']['grayscale'],
                        device=image_matching_device
                    )
                    loftr_output = loftr.match_images(
                        image1=image1,
                        image2=image2,
                        model=loftr_model,
                        device=image_matching_device,
                        amp=config['inference']['image_matching']['amp'],
                        confidence_threshold=0.6
                    )

                    fundamental_matrix, inliers = cv2.findFundamentalMat(
                        points1=loftr_output['keypoints1'],
                        points2=loftr_output['keypoints2'],
                        method=cv2.USAC_MAGSAC,
                        ransacReprojThreshold=0.25,
                        confidence=0.99999,
                        maxIters=100000
                    )
                    inliers = inliers.reshape(-1).astype(bool)

                    if config['inference']['image_matching']['visualize']:
                        image1 = image_utilities.get_image_tensor(
                            image_path_or_array=image_paths[image_1_idx],
                            resize_shape=config['transforms']['image_matching']['resize_shape'],
                            resize_longest_edge=config['transforms']['image_matching']['resize_longest_edge'],
                            grayscale=False,
                            device=image_matching_device
                        )
                        image2 = image_utilities.get_image_tensor(
                            image_path_or_array=image_paths[image_2_idx],
                            resize_shape=config['transforms']['image_matching']['resize_shape'],
                            resize_longest_edge=config['transforms']['image_matching']['resize_longest_edge'],
                            grayscale=False,
                            device=image_matching_device
                        )
                        visualization.visualize_image_matching(
                            image1=image1, image2=image2,
                            keypoints1=loftr_output['keypoints1'],
                            keypoints2=loftr_output['keypoints2'],
                            inliers=inliers,
                            path=None
                        )

    elif args.mode == 'submission':
        pass

    else:
        raise ValueError(f'Invalid mode: {args.mode}')