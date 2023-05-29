import torch

import image_utilities


def extract_keypoints_and_descriptors(image, model, model_parameters, amp, device):

    """
    Extract keypoints and descriptors on given image with given model

    Parameters
    ----------
    image: torch.Tensor of shape (1, 3, height, width)
        Image tensor

    model: torch.nn.Module
        DISK model

    model_parameters: dict
        DISK model parameters

    amp: bool
        Whether to use auto mixed precision or not

    device: torch.device
        Location of the image1, image2 and the model

    Returns
    -------
    descriptors: torch.Tensor of shape (n_detections, n_dimensions)
        Local descriptors

    keypoints: numpy.ndarray of shape (n_detections, 2)
        Keypoints
    """

    with torch.no_grad():
        if amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                disk_features = model(image, **model_parameters)
        else:
            disk_features = model(image, **model_parameters)

    keypoints = disk_features[0].keypoints.detach().cpu().numpy()
    descriptors = disk_features[0].descriptors.detach().cpu()

    return descriptors, keypoints


def match_descriptors(descriptors1, descriptors2, matcher):

    """
    Match descriptors with nearest neighbor algorithm

    Parameters
    ----------
    descriptors1: torch.Tensor of shape (n_detections, n_dimensions)
        Descriptors from first image

    descriptors2: torch.Tensor of shape (n_detections, n_dimensions):
        Descriptors from second image

    matcher: torch.nn.Module
        Descriptor matcher

    Returns
    -------
    distances: numpy.ndarray of shape (n_matches)
        Distances of matching descriptors

    indexes: numpy.ndarray of shape (n_matches, 2)
        Indexes of matching descriptors
    """

    with torch.no_grad():
        distances, indexes = matcher(descriptors1, descriptors2)

    distances = distances.detach().cpu().numpy().reshape(-1)
    indexes = indexes.detach().cpu().numpy()

    return distances, indexes


def match_images(image1, image2, model, model_parameters, matcher, device, amp, transforms, distance_threshold, top_k):

    """
    Match given two images with each other using given model and matcher

    Parameters
    ----------
    image1: numpy.ndarray of shape (3, height, width)
        Array of first image

    image2: numpy.ndarray of shape (3, height, width)
        Array of second image

    model: torch.nn.Module
        DISK model

    model_parameters: dict
        DISK model parameters

    matcher: torch.nn.Module
        Descriptor matcher

    device: torch.device
        Location of the image1, image2 and the model

    amp: bool
        Whether to use auto mixed precision or not

    transforms: dict
        Dictionary of transform parameters

    distance_threshold: float
        Threshold to filter out keypoints with low distance

    top_k: int
        Number of keypoints to take

    Returns
    -------
    outputs: dict
        Model outputs
    """

    image1_raw_height, image1_raw_width = image1.shape[:2]
    image1 = image_utilities.get_image_tensor(
        image_path_or_array=image1,
        resize=transforms['resize'],
        resize_shape=transforms['resize_shape'],
        resize_longest_edge=transforms['resize_longest_edge'],
        scale=transforms['scale'],
        grayscale=transforms['grayscale']
    )
    image1 = image1.to(device)
    image1_transformed_height, image1_transformed_width = image1.shape[2:]

    image2_raw_height, image2_raw_width = image2.shape[:2]
    image2 = image_utilities.get_image_tensor(
        image_path_or_array=image2,
        resize=transforms['resize'],
        resize_shape=transforms['resize_shape'],
        resize_longest_edge=transforms['resize_longest_edge'],
        scale=transforms['scale'],
        grayscale=transforms['grayscale']
    )
    image2 = image2.to(device)
    image2_transformed_height, image2_transformed_width = image2.shape[2:]

    descriptors1, keypoints1 = extract_keypoints_and_descriptors(image=image1, model=model, model_parameters=model_parameters, amp=amp, device=device)
    descriptors2, keypoints2 = extract_keypoints_and_descriptors(image=image2, model=model, model_parameters=model_parameters, amp=amp, device=device)
    distances, indexes = match_descriptors(descriptors1=descriptors1, descriptors2=descriptors2, matcher=matcher)

    outputs = {
        'keypoints0': keypoints1[indexes[:, 0]],
        'keypoints1': keypoints2[indexes[:, 1]],
        'distances': distances
    }

    if distance_threshold is not None:
        if isinstance(distance_threshold, float):
            # Select matched keypoints with above given distance threshold
            distance_mask = outputs['distances'] >= distance_threshold
        elif isinstance(distance_threshold, int):
            # Select keypoints dynamically based on distance distribution
            distance_mean, distance_std = outputs['distances'].mean(), outputs['distances'].std()
            distance_mask = outputs['distances'] >= (distance_mean + (distance_std * distance_threshold))
        else:
            raise ValueError(f'Invalid distance_threshold {distance_threshold}')

        for k in outputs.keys():
            outputs[k] = outputs[k][distance_mask]

    if top_k is not None:
        # Select top-k keypoints based on their distances
        sorting_idx = outputs['distances'].argsort()[-top_k:]
        for k in outputs.keys():
            outputs[k] = outputs[k][sorting_idx]

    outputs['keypoints0'][:, 0] *= image1_raw_width / image1_transformed_width
    outputs['keypoints0'][:, 1] *= image1_raw_height / image1_transformed_height
    outputs['keypoints1'][:, 0] *= image2_raw_width / image2_transformed_width
    outputs['keypoints1'][:, 1] *= image2_raw_height / image2_transformed_height

    return outputs
