import numpy as np


def rotation_matrix_to_quaternion(rotation_matrix):

    """
    Convert rotation matrix to quaternion

    Parameters
    ----------
    rotation_matrix: numpy.ndarray of shape (3, 3)
        Array of directions of the world-axes in camera coordinates

    Returns
    -------
    quaternion: numpy.ndarray of shape (4)
        Array of quaternion
    """

    r00 = rotation_matrix[0, 0]
    r01 = rotation_matrix[0, 1]
    r02 = rotation_matrix[0, 2]
    r10 = rotation_matrix[1, 0]
    r11 = rotation_matrix[1, 1]
    r12 = rotation_matrix[1, 2]
    r20 = rotation_matrix[2, 0]
    r21 = rotation_matrix[2, 1]
    r22 = rotation_matrix[2, 2]

    k = np.array([
        [r00 - r11 - r22, 0.0, 0.0, 0.0],
        [r01 + r10, r11 - r00 - r22, 0.0, 0.0],
        [r02 + r20, r12 + r21, r22 - r00 - r11, 0.0],
        [r21 - r12, r02 - r20, r10 - r01, r00 + r11 + r22]
    ])
    k /= 3.0

    # Quaternion is the eigenvector of k that corresponds to the largest eigenvalue
    w, v = np.linalg.eigh(k)
    quaternion = v[[3, 0, 1, 2], np.argmax(w)]

    if quaternion[0] < 0:
        np.negative(quaternion, quaternion)

    return quaternion
