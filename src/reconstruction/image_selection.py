import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def select_images(image_paths, image_selection_features, image_count):

    """
    Select most similar images

    Parameters
    ----------
    image_paths: list of shape (n_images)
        List of image paths

    image_selection_features: np.ndarray of shape (n_images, n_features)
        Features array

    image_count: int
        Image count to retrieve most similar images

    Returns
    -------
    image_paths: list of shape (image_count)
        List of most similar image paths
    """

    # Calculate pairwise cosine similarities between features
    pairwise_cosine_similarities = cosine_similarity(image_selection_features)

    # Zero the diagonal and calculate mean cosine similarities
    np.fill_diagonal(pairwise_cosine_similarities, 0)
    mean_cosine_similarities = pairwise_cosine_similarities.mean(axis=1)

    # Extract sorting index in descending order
    sorting_idx = np.argsort(mean_cosine_similarities)[::-1]

    image_paths = np.array(image_paths)
    image_paths = image_paths[sorting_idx][:image_count].tolist()

    return image_paths
