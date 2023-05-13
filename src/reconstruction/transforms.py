import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def create_image_selection_transforms(**transform_parameters):

    """
    Create transformation pipeline for image selection

    Parameters
    ----------
    transform_parameters: dict
        Dictionary of transform parameters

    Returns
    -------
    transforms: dict
        Transform pipeline for image selection
    """

    image_selection_transforms = A.Compose([
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_NEAREST,
            always_apply=True
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    return image_selection_transforms
