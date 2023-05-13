import numpy as np
import cv2
from torch.utils.data import Dataset


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
        images: torch.FloatTensor of shape (channel, height, width) or numpy.ndarray of shape (height, width, channel)
            Image tensor or array
        """

        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=np.array(image))['image'].float()

        return image
