import torch
from kornia.feature import (
    LocalFeature, PassLAF, LAFOrienter, PatchDominantGradientOrientation, OriNet, LAFAffNetShapeEstimator,
    KeyNetDetector, LAFDescriptor, HardNet8, HyNet, TFeat, SOSNet, get_laf_center
)

import image_utilities


class LocalFeatureDetectorDescriptor(LocalFeature):

    def __init__(
            self,
            orientation_module_name, orientation_module_parameters, orientation_module_weights_path,
            affine_module_name, affine_module_parameters, affine_module_weights_path,
            detector_module_name, detector_module_parameters, detector_module_weights_path,
            descriptor_module_name, descriptor_module_parameters, descriptor_module_weights_path,
    ):

        """
        Module that combines local feature detector and descriptor

        Parameters
        ----------
        orientation_module_name: str
            Name of the orientation module

        orientation_module_parameters: dict
            Parameters of the orientation module

        orientation_module_weights_path: str
            Path of the orientation module weights

        affine_module_name: str
            Name of the affine module

        affine_module_parameters: dict
            Parameters of the affine module

        affine_module_weights_path: str
            Path of the affine module weights

        detector_module_name: str
            Name of the detector module

        detector_module_parameters: dict
            Parameters of the detector module

        detector_module_weights_path: str
            Path of the detector module weights

        descriptor_module_name: str
            Name of the descriptor module

        descriptor_module_parameters: dict
            Parameters of the descriptor module

        descriptor_module_weights_path: str
            Path of the descriptor module weights

        Returns
        -------
        lafs: torch.Tensor of shape (1, n_detections, 2, 3)
            Detected local affine frames

        responses: torch.Tensor of shape (1, n_detections)
            Response function values for corresponding lafs

        descriptors: torch.Tensor of shape (1, n_detections, n_dimensions)
            Local descriptors
        """

        # Instantiate specified orientation module
        if orientation_module_name == 'PassLAF':
            orientation_module = PassLAF()
        elif orientation_module_name == 'OriNet':
            orientation_module = LAFOrienter(**orientation_module_parameters, angle_detector=OriNet(pretrained=(orientation_module_weights_path is None)))
        elif orientation_module_name == 'PatchDominantGradientOrientation':
            orientation_module = LAFOrienter(**orientation_module_parameters, angle_detector=PatchDominantGradientOrientation())
        else:
            orientation_module = None

        # Instantiate specified affine module
        if affine_module_name == 'LAFAffNetShapeEstimator':
            affine_module = LAFAffNetShapeEstimator(**affine_module_parameters, pretrained=(affine_module_weights_path is None))
        else:
            affine_module = None

        # Instantiate specified detector module
        if detector_module_name == 'KeyNetDetector':
            detector_module = KeyNetDetector(**detector_module_parameters, pretrained=(detector_module_weights_path is None), ori_module=orientation_module, aff_module=affine_module)
        else:
            detector_module = None

        # Load pretrained weights for the detector module
        if orientation_module_weights_path is not None:
            detector_module.ori.angle_detector.load_state_dict(torch.load(orientation_module_weights_path)['state_dict'])
        if affine_module_weights_path is not None:
            detector_module.aff.load_state_dict(torch.load(affine_module_weights_path)['state_dict'])
        if detector_module_weights_path is not None:
            detector_module.model.load_state_dict(torch.load(detector_module_weights_path)['state_dict'])

        # Instantiate specified descriptor module
        if descriptor_module_name == 'HardNet8':
            descriptor_module = LAFDescriptor(**descriptor_module_parameters, patch_descriptor_module=HardNet8(pretrained=(descriptor_module_weights_path is None)))
        elif descriptor_module_name == 'HyNet':
            descriptor_module = LAFDescriptor(**descriptor_module_parameters, patch_descriptor_module=HyNet(pretrained=(descriptor_module_weights_path is None)))
        elif descriptor_module_name == 'TFeat':
            descriptor_module = LAFDescriptor(**descriptor_module_parameters, patch_descriptor_module=TFeat(pretrained=(descriptor_module_weights_path is None)))
        elif descriptor_module_name == 'SOSNet':
            descriptor_module = LAFDescriptor(**descriptor_module_parameters, patch_descriptor_module=SOSNet(pretrained=(descriptor_module_weights_path is None)))
        else:
            descriptor_module = None

        # Load pretrained weights for the descriptor module
        if descriptor_module_weights_path is not None:
            descriptor_module.descriptor.load_state_dict(torch.load(descriptor_module_weights_path))

        super().__init__(detector_module, descriptor_module)


def extract_keypoints_and_descriptors(image, model, amp, device):

    """
    Extract local feature descriptors on given image with given model

    Parameters
    ----------
    image: torch.Tensor of shape (1, 1, height, width)
        Image tensor

    model: torch.nn.Module
        Local feature detector and descriptor model

    device: torch.device
        Location of the image1, image2 and the model

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    lafs: numpy.ndarray of shape (n_detections, 2, 3)
        Detected local affine frames

    responses: numpy.ndarray of shape (n_detections)
        Response function values for corresponding lafs

    descriptors: numpy.ndarray of shape (n_detections, n_dimensions)
        Local descriptors

    keypoints: numpy.ndarray of shape (n_detections, 2)
        Keypoints
    """

    with torch.no_grad():
        if amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                lafs, responses, descriptors = model(image)
        else:
            lafs, responses, descriptors = model(image)

    responses = torch.squeeze(responses, dim=0).detach().cpu().numpy()
    descriptors = torch.squeeze(descriptors, dim=0).detach().cpu()
    keypoints = get_laf_center(lafs)
    keypoints = keypoints.detach().cpu().numpy().reshape(-1, 2)
    lafs = lafs.detach().cpu().numpy()

    return lafs, responses, descriptors, keypoints


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


def match_images(image1, image2, model, matcher, device, amp, transforms, distance_threshold, top_k):

    """
    Match given two images with each other using given model and matcher

    Parameters
    ----------
    image1: numpy.ndarray of shape (3, height, width)
        Array of first image

    image2: numpy.ndarray of shape (3, height, width)
        Array of second image

    model: torch.nn.Module
        Local feature detector and descriptor model

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

    _, _, descriptors1, keypoints1 = extract_keypoints_and_descriptors(image=image1, model=model, amp=amp, device=device)
    _, _, descriptors2, keypoints2 = extract_keypoints_and_descriptors(image=image2, model=model, amp=amp, device=device)
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
