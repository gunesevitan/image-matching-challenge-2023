import torch

import image_utilities


def match_images(image1, image2, model, device, amp, transforms, confidence_threshold):

    """
    Match given two images with each other using LoFTR model

    Parameters
    ----------
    image1: numpy.ndarray of shape (3, height, width)
        Array of first image

    image2: numpy.ndarray of shape (3, height, width)
        Array of second image

    model: torch.nn.Module
        LoFTR Model

    device: torch.device
        Location of the image1, image2 and the model

    amp: bool
        Whether to use auto mixed precision or not

    transforms: dict
        Dictionary of transform parameters

    confidence_threshold: float or int
        Confidence threshold

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

    inputs = {
        'image0': image1,
        'image1': image2
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

    if confidence_threshold is not None:
        if isinstance(confidence_threshold, float):
            # Select keypoints above given confidence threshold
            confidence_mask = outputs['confidence'] >= confidence_threshold
        elif isinstance(confidence_threshold, int):
            # Select keypoints dynamically based on confidence distribution
            confidence_mean, confidence_std = outputs['confidence'].mean(), outputs['confidence'].std()
            confidence_mask = outputs['confidence'] >= (confidence_mean + (confidence_std * confidence_threshold))
        else:
            raise ValueError(f'Invalid confidence_threshold {confidence_threshold}')

        outputs = {
            'keypoints0': outputs['keypoints0'][confidence_mask],
            'keypoints1': outputs['keypoints1'][confidence_mask],
            'confidence': outputs['confidence'][confidence_mask],
            'batch_indexes': outputs['batch_indexes'][confidence_mask]
        }

    outputs['keypoints0'][:, 0] *= image1_raw_width / image1_transformed_width
    outputs['keypoints0'][:, 1] *= image1_raw_height / image1_transformed_height
    outputs['keypoints1'][:, 0] *= image2_raw_width / image2_transformed_width
    outputs['keypoints1'][:, 1] *= image2_raw_height / image2_transformed_height

    return outputs
