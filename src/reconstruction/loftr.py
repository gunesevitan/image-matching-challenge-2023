import torch

import image_utilities


def match_images(image1, image2, model, device, amp, transforms):

    """
    Match given two images with each other using LoFTR model

    Parameters
    ----------
    image1: numpy.ndarray of shape (3, height, width)
        Batch of first images tensor

    image2: numpy.ndarray of shape (3, height, width)
        Batch of second images tensor

    model: torch.nn.Module
        LoFTR Model

    device: torch.device
        Location of the image1, image2 and the model

    amp: bool
        Whether to use auto mixed precision or not

    transforms: dict
        Dictionary of transform parameters

    Returns
    -------
    outputs: dict
        Model outputs
    """

    image1_raw_height, image1_raw_width = image1.shape[:2]
    image1 = image_utilities.get_image_tensor(
        image_path_or_array=image1,
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

    outputs['keypoints0'][:, 0] *= image1_raw_width / image1_transformed_width
    outputs['keypoints0'][:, 1] *= image1_raw_height / image1_transformed_height
    outputs['keypoints1'][:, 0] *= image2_raw_width / image2_transformed_width
    outputs['keypoints1'][:, 1] *= image2_raw_height / image2_transformed_height

    return outputs
