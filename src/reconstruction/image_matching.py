import h5py
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

import datasets


def create_image_pairs(image_paths):

    """
    Create all possible image pairs from given list of image paths

    Parameters
    ----------
    image_paths: list of shape (n_images)
        List of image paths

    Returns
    -------
    image_pair_indices: list of shape (n_image_pairs)
        List of tuples of image pair indices
    """

    image_pair_indices = []

    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            image_pair_indices.append((i, j))

    return image_pair_indices


def prepare_dataloader(image_paths, image_pair_indices, transforms, batch_size, num_workers):

    """
    Prepare data loader for inference

    Parameters
    ----------
    image_paths: list of shape (n_images)
        List of image paths

    image_pair_indices: list of shape (n_image_pairs)
        List of tuples of image pair indices

    transforms: dict
        Transform pipeline

    batch_size: int
        Batch size of the data loader

    num_workers: int
        Number of workers of the data loader

    Returns
    -------
    data_loader: torch.utils.data.DataLoader
        Data loader
    """

    dataset = datasets.ImagePairDataset(
        image_paths=image_paths,
        image_pair_indices=image_pair_indices,
        transforms=transforms
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers
    )

    return data_loader


def match_images(first_images, second_images, model, device, amp):

    """
    Match given two images with each other using LoFTR model

    Parameters
    ----------
    first_images: torch.Tensor of shape (batch_size, 1, height, width)
        Batch of first images tensor

    second_images: torch.Tensor of shape (batch_size, 1, height, width)
        Batch of second images tensor

    model: torch.nn.Module
        LoFTR Model

    device: torch.device
        Location of the first_images, second_images and the model

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    outputs: dict of {
        keypoints1: numpy.ndarray of shape (n_keypoints, 2)
        keypoints2: numpy.ndarray of shape (n_keypoints, 2)
        confidences: numpy.ndarray of shape (n_keypoints)
        batch_indexes: numpy.ndarray of shape (n_keypoints)
    }
        Matched keypoints from first and second images, confidences and batch indexes
    """

    inputs = {
        'image0': first_images,
        'image1': second_images
    }

    with torch.no_grad():
        if amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(inputs)
        else:
            outputs = model(inputs)

    outputs = {
        'keypoints0': outputs['keypoints0'].cpu().numpy(),
        'keypoints1': outputs['keypoints1'].cpu().numpy(),
        'confidence': outputs['confidence'].cpu().numpy(),
        'batch_indexes': outputs['batch_indexes'].cpu().numpy()
    }

    return outputs


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

    with h5py.File(output_directory / 'loftr_matches.h5', mode='w') as f:
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

    with h5py.File(output_directory / 'loftr_matches.h5', mode='r') as f:
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
                unique_keypoints[first_image_filename][image_pair_match_index_copy[:, 0]],
                unique_keypoints[second_image_filename][image_pair_match_index_copy[:, 1]]
            ], axis=1)

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
