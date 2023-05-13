import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    df = pd.read_csv(settings.DATA / 'train' / 'train_labels.csv')
    settings.logger.info(f'Dataset Shape: {df.shape}')

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        image_path = settings.DATA / 'train' / row['image_path']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        df.loc[idx, 'image_height'] = image.shape[0]
        df.loc[idx, 'image_width'] = image.shape[1]

        df.loc[idx, 'image_r_mean'] = np.mean(image[:, :, 0])
        df.loc[idx, 'image_r_std'] = np.std(image[:, :, 0])
        df.loc[idx, 'image_g_mean'] = np.mean(image[:, :, 1])
        df.loc[idx, 'image_g_std'] = np.std(image[:, :, 1])
        df.loc[idx, 'image_b_mean'] = np.mean(image[:, :, 2])
        df.loc[idx, 'image_b_std'] = np.std(image[:, :, 2])

    df.to_csv(settings.DATA / 'train' / 'train_metadata.csv', index=False)
    settings.logger.info(f'Saved train_metadata.csv to {settings.DATA / "train"}')
