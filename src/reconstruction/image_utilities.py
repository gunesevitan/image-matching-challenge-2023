import pathlib
import cv2
import kornia


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


def resize_with_aspect_ratio(image, longest_edge):

    """
    Resize image while preserving its aspect ratio

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, 3)
        Image array

    longest_edge: int
        Desired number of pixels on the longest edge

    Returns
    -------
    image: numpy.ndarray of shape (resized_height, resized_width, 3)
        Resized image array
    """

    height, width = image.shape[:2]
    scale = longest_edge / max(height, width)
    image = cv2.resize(image, dsize=(int(width * scale), int(height * scale)), interpolation=cv2.INTER_LANCZOS4)

    return image


def get_image_tensor(image_path_or_array, resize, resize_shape, resize_longest_edge, scale, grayscale):

    """
    Load image and preprocess it

    Parameters
    ----------
    image_path_or_array: str or numpy.ndarray of shape (height, width, 3)
        Image path or image array

    resize: bool
        Whether to resize the image or not

    resize_shape: tuple or int
        Tuple of image height and width or number of pixels for both height and width

    resize_longest_edge: bool
        Whether to resize the longest edge or not

    scale: bool
        Whether to scale image pixel values by max 8-bit pixel value or not

    grayscale: bool
        Whether to convert RGB image to grayscale or not

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

    if resize:
        if resize_longest_edge:
            image = resize_with_aspect_ratio(image=image, longest_edge=resize_shape)
        else:
            resize_shape = (resize_shape, resize_shape) if isinstance(resize_shape, int) else resize_shape
            image = cv2.resize(image, resize_shape, interpolation=cv2.INTER_LANCZOS4)

    if scale:
        image = image / 255.

    image = kornia.image_to_tensor(image, False).float()
    if grayscale:
        image = kornia.color.rgb_to_grayscale(image)

    return image
