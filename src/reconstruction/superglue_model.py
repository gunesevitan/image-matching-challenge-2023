import numpy as np
import pandas as pd
import torch

import image_utilities
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


def make_crop(image, keypoints, perc_points=0.85, pad=5):
    norm_keypoints = MinMaxScaler().fit_transform(keypoints)
    total = len(keypoints)
    best_dist = 1
    best_clusters = None
    best_asm = None
    for eps in [0.01, 0.025, 0.05, 0.1, 0.2]:
        clusters = DBSCAN(eps=eps).fit_predict(norm_keypoints)
        counts = pd.Series(clusters).value_counts().sort_values(ascending=False)
        counts = counts[counts.index > -1]
        if len(counts) == 0:
            continue

        cumsums = np.cumsum(counts.values) / total
        dists = np.abs(cumsums - perc_points)
        best_ix = np.argmin(dists)

        if dists[best_ix] < best_dist:
            best_dist = dists[best_ix]
            best_clusters = list(counts.head(best_ix + 1).index)
            best_asm = clusters

    mask = np.isin(best_asm, best_clusters)

    miny = int(np.min(keypoints[mask][:, 1]))
    miny = max(miny - pad, 0)

    maxy = int(np.max(keypoints[mask][:, 1]))
    maxy = min(maxy + pad, image.shape[0])

    minx = int(np.min(keypoints[mask][:, 0]))
    minx = max(minx - pad, 0)

    maxx = int(np.max(keypoints[mask][:, 0]))
    maxx = min(maxx + pad, image.shape[1])

    #keypoints[:, 0] -= minx
    #keypoints[:, 1] -= miny

    return image[miny:maxy + 1, minx:maxx + 1, :], minx, miny


def match_images(image1, image2, model, device, amp, transforms, score_threshold, top_k):

    """
    Match given two images with each other using SuperGlue model

    Parameters
    ----------
    image1: numpy.ndarray of shape (3, height, width)
        Array of first image

    image2: numpy.ndarray of shape (3, height, width)
        Array of second image

    model: torch.nn.Module
        SuperGlue Model

    device: torch.device
        Location of the image1, image2 and the model

    amp: bool
        Whether to use auto mixed precision or not

    transforms: dict
        Dictionary of transform parameters

    score_threshold: float or int
        Confidence threshold

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
                outputs = model({'image0': image1, 'image1': image2})
        else:
            outputs = model({'image0': image1, 'image1': image2})

    for k in outputs.keys():
        if k == 'descriptors0' or k == 'descriptors1':
            outputs[k] = outputs[k][0].detach().cpu().numpy().T
        else:
            outputs[k] = outputs[k][0].detach().cpu().numpy()

    matches_mask = outputs['matches0'] > -1

    for k in ['keypoints1', 'scores1', 'descriptors1', 'matches1', 'matching_scores1']:
        outputs[k] = outputs[k][outputs['matches0'][matches_mask]]

    for k in ['keypoints0', 'scores0', 'descriptors0', 'matches0', 'matching_scores0']:
        outputs[k] = outputs[k][matches_mask]

    if score_threshold is not None:
        if isinstance(score_threshold, float):
            # Select matched keypoints with above given score threshold
            score_mask = outputs['matching_scores0'] >= score_threshold
        elif isinstance(score_threshold, int):
            # Select keypoints dynamically based on score distribution
            score_mean, score_std = outputs['matching_scores0'].mean(), outputs['matching_scores0'].std()
            score_mask = outputs['matching_scores0'] >= (score_mean + (score_std * score_threshold))
        else:
            raise ValueError(f'Invalid score_threshold {score_threshold}')

        for k in outputs.keys():
            outputs[k] = outputs[k][score_mask]

    if top_k is not None:
        # Select top-k keypoints based on their scores
        sorting_idx = outputs['matching_scores0'].argsort()[-top_k:]
        for k in outputs.keys():
            outputs[k] = outputs[k][sorting_idx]

    outputs['keypoints0'][:, 0] *= image1_raw_width / image1_transformed_width
    outputs['keypoints0'][:, 1] *= image1_raw_height / image1_transformed_height
    outputs['keypoints1'][:, 0] *= image2_raw_width / image2_transformed_width
    outputs['keypoints1'][:, 1] *= image2_raw_height / image2_transformed_height

    return outputs
