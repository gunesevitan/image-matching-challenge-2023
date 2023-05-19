import numpy as np
import cv2
from torch.utils.data import Dataset

import image_utilities


class ImageDataset(Dataset):

    def __init__(self, image_paths, transforms=None):

        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.image_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        image: torch.FloatTensor of shape (channel, height, width) or numpy.ndarray of shape (height, width, channel)
            Image tensor or array
        """

        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=np.array(image))['image'].float()

        return image


class ImagePairDataset(Dataset):

    def __init__(self, image_paths, image_pair_indices, transforms):

        self.image_paths = image_paths
        self.image_pair_indices = image_pair_indices
        self.transforms = transforms

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.image_pair_indices)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        image1: torch.FloatTensor of shape (channel, height, width)
            First image tensor

        image2: torch.FloatTensor of shape (channel, height, width)
            Second image tensor
        """

        image1 = image_utilities.get_image_tensor(
            image_path_or_array=str(self.image_paths[self.image_pair_indices[idx][0]]),
            resize_shape=self.transforms['resize_shape'],
            resize_longest_edge=self.transforms['resize_longest_edge'],
            scale=self.transforms['scale'],
            grayscale=self.transforms['grayscale']
        )

        image2 = image_utilities.get_image_tensor(
            image_path_or_array=str(self.image_paths[self.image_pair_indices[idx][1]]),
            resize_shape=self.transforms['resize_shape'],
            resize_longest_edge=self.transforms['resize_longest_edge'],
            scale=self.transforms['scale'],
            grayscale=self.transforms['grayscale']
        )

        return image1, image2
