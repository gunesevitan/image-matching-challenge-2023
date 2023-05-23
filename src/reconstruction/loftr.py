import torch
from torch.utils.data import DataLoader, SequentialSampler

import datasets


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
        'keypoints0': outputs['keypoints0'].detach().cpu().numpy(),
        'keypoints1': outputs['keypoints1'].detach().cpu().numpy(),
        'confidence': outputs['confidence'].detach().cpu().numpy(),
        'batch_indexes': outputs['batch_indexes'].detach().cpu().numpy()
    }

    return outputs
