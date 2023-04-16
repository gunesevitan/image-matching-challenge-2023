import pathlib
import cv2
import torch
import kornia
from kornia.feature import laf_from_center_scale_ori
from kornia_moons.feature import draw_LAF_matches
import matplotlib.pyplot as plt


def visualize_image(image_path_or_array, path=None):

    """
    Visualize image

    Parameters
    ----------
    image_path_or_array: str or numpy.ndarray of shape (height, width, 3)
        Image path or image array

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if isinstance(image_path_or_array, pathlib.Path) or isinstance(image_path_or_array, str):
        # Read image from the given path if image_path_or_array is a path-like string
        image = cv2.imread(str(image_path_or_array))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path_or_array

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_image_pair(image_path_or_array1, image_path_or_array2, path=None):

    """
    Visualize image

    Parameters
    ----------
    image_path_or_array1: str or numpy.ndarray of shape (height, width, 3)
        First image path or image array

    image_path_or_array2: str or numpy.ndarray of shape (height, width, 3)
        Second image path or image array

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if isinstance(image_path_or_array1, pathlib.Path) or isinstance(image_path_or_array1, str):
        # Read image from the given path if image_path_or_array1 is a path-like string
        image1 = cv2.imread(str(image_path_or_array1))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    else:
        image1 = image_path_or_array1

    if isinstance(image_path_or_array2, pathlib.Path) or isinstance(image_path_or_array2, str):
        # Read image from the given path if image_path_or_array2 is a path-like string
        image2 = cv2.imread(str(image_path_or_array2))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    else:
        image2 = image_path_or_array2

    fig, axes = plt.subplots(figsize=(16, 32), ncols=2)
    axes[0].imshow(image1)
    axes[1].imshow(image2)

    for i in range(2):
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_image_matching(image1, image2, keypoints1, keypoints2, inliers, path=None):

    """
    Visualize matched keypoints between an image pair

    Parameters
    ----------
    image1: torch.Tensor of shape (1, 3, height, width)
        First image tensor

    image2: torch.Tensor of shape (1, 3, height, width)
        Second image tensor

    keypoints1: numpy.ndarray of shape (n_keypoints, 2)
        Keypoints from first image

    keypoints2: numpy.ndarray of shape (n_keypoints, 2)
        Keypoints from second image

    inliers: numpy.ndarray of shape (n_keypoints)
        Inlier mask

    path: path-like str or None
        Path of the output file (if path is None, plot is displayed with selected backend)
    """

    draw_LAF_matches(
        lafs1=laf_from_center_scale_ori(
            torch.from_numpy(keypoints1).view(1, -1, 2),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1)
        ),
        lafs2=laf_from_center_scale_ori(
            torch.from_numpy(keypoints2).view(1, -1, 2),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1)
         ),
        tent_idxs=torch.arange(keypoints1.shape[0]).view(-1, 1).repeat(1, 2),
        img1=kornia.tensor_to_image(image1),
        img2=kornia.tensor_to_image(image2),
        inlier_mask=inliers,
        draw_dict={
            'inlier_color': (0.2, 1, 0.2),
            'tentative_color': None,
            'feature_color': (0.2, 0.5, 1),
            'vertical': False
        }
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
