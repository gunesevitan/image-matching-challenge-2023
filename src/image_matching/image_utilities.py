import pathlib
import cv2
import kornia


def resize_with_aspect_ratio(image, longest_edge):

    """
    Resize image while preserving its aspect ratio

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, 3)
        Image array

    longest_edge: int
        Number of pixels on the longest edge

    Returns
    -------
    image: numpy.ndarray of shape (resized_height, resized_width, 3)
        Resized image array
    """

    height, width = image.shape[:2]
    scale = longest_edge / max(height, width)
    image = cv2.resize(image, dsize=(int(width * scale), int(height * scale)), interpolation=cv2.INTER_NEAREST)

    return image


def get_image_tensor(image_path_or_array, resize_shape, resize_longest_edge, grayscale, device):

    """
    Load image and move it to specified device

    Parameters
    ----------
    image_path_or_array: str or numpy.ndarray of shape (height, width, 3)
        Image path or image array

    resize_shape: tuple or int
        Tuple of image height and width or number of pixels for both height and width

    resize_longest_edge: bool
        Whether to resize longest edge or not

    grayscale: bool
        Whether to convert RGB image to grayscale

    device: torch.device
        Location of the image tensor

    Returns
    -------
    image: torch.Tensor of shape (1, 1 or 3, height, width)
        Image tensor
    """

    if isinstance(image_path_or_array, pathlib.Path) or isinstance(image_path_or_array, str):
        # Read image from the given path if image_path_or_array is a path-like string
        image = cv2.imread(str(image_path_or_array))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path_or_array

    if resize_longest_edge:
        image = resize_with_aspect_ratio(image=image, longest_edge=resize_shape)
    else:
        image = cv2.resize(image, dsize=resize_shape, interpolation=cv2.INTER_NEAREST)

    image = image / 255.

    image = kornia.image_to_tensor(image, False).float()
    if grayscale:
        image = kornia.color.rgb_to_grayscale(image)

    image = image.to(device)

    return image