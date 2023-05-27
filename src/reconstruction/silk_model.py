import sys
import torch

import image_utilities
sys.path.append('..')
import settings
sys.path.append(str(settings.ROOT / 'venv' / 'lib' / 'python3.9' / 'site-packages' / 'silk'))
import silk
from silk.backbones.silk.silk import from_feature_coords_to_image_coords


def match_images(image1, image2, model, device, amp, transforms, top_k):

    """
    Match given two images with each other using SiLK model

    Parameters
    ----------
    image1: numpy.ndarray of shape (3, height, width)
        Batch of first images tensor

    image2: numpy.ndarray of shape (3, height, width)
        Batch of second images tensor

    model: torch.nn.Module
        SiLK Model

    device: torch.device
        Location of the image1, image2 and the model

    amp: bool
        Whether to use auto mixed precision or not

    transforms: dict
        Dictionary of transform parameters

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

    with torch.no_grad():
        if amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                keypoints1, descriptors1, _ = model(image1)
        else:
            keypoints1, descriptors1, _ = model(image1)

    keypoints1 = from_feature_coords_to_image_coords(model, keypoints1).reshape(-1, 3)
    keypoints1 *= image1_raw_width / image1_transformed_width
    descriptors1 = descriptors1.reshape(1, 128, -1).permute(0, 2, 1)
    descriptors1 = torch.squeeze(descriptors1, dim=0)

    keypoints1 = keypoints1.detach().cpu().numpy()
    descriptors1 = descriptors1.detach().cpu().numpy()

    if top_k is not None:
        # Select top-k keypoints based on their scores
        sorting_idx = keypoints1[:, 2].argsort()[-top_k:]
        keypoints1 = keypoints1[sorting_idx]
        descriptors1 = descriptors1[sorting_idx]

    with torch.no_grad():
        if amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                keypoints2, descriptors2, _ = model(image2)
        else:
            keypoints2, descriptors2, _ = model(image2)

    keypoints2 = from_feature_coords_to_image_coords(model, keypoints2).reshape(-1, 3)
    keypoints2 *= image2_raw_width / image2_transformed_width
    descriptors2 = descriptors2.reshape(1, 128, -1).permute(0, 2, 1)
    descriptors2 = torch.squeeze(descriptors2, dim=0)

    keypoints2 = keypoints2.detach().cpu().numpy()
    descriptors2 = descriptors2.detach().cpu().numpy()

    if top_k is not None:
        # Select top-k keypoints based on their scores
        sorting_idx = keypoints2[:, 2].argsort()[-top_k:]
        keypoints2 = keypoints2[sorting_idx]
        descriptors2 = descriptors2[sorting_idx]

    keypoints1[:, 0] *= image1_raw_height / image1_transformed_height
    keypoints1[:, 1] *= image1_raw_width / image1_transformed_width
    keypoints2[:, 0] *= image2_raw_height / image2_transformed_height
    keypoints2[:, 1] *= image2_raw_width / image2_transformed_width

    matcher = silk.models.silk.matcher(
        postprocessing='double-softmax',
        threshold=0.99,
        temperature=0.1,
    )

    with torch.no_grad():
        matches = matcher(torch.as_tensor(descriptors1), torch.as_tensor(descriptors2))
        matches = matches.detach().cpu().numpy()

    keypoints1 = keypoints1[:, [1, 0]][matches[:, 0]]
    keypoints2 = keypoints2[:, [1, 0]][matches[:, 1]]

    outputs = {
        'keypoints0': keypoints1,
        'keypoints1': keypoints2
    }

    return outputs
