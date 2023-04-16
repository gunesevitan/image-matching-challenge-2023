def create_all_image_pairs(image_paths):

    """
    Create all possible image pairs from given list of image_paths

    Parameters
    ----------
    image_paths: list of shape (n_image_paths)
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
