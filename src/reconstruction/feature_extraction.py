import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import timm

import datasets


def load_feature_extractor(model_name, pretrained, model_args):

    """
    Load specified pretrained model for feature extraction

    Parameters
    ----------
    model_name: str
        Model name

    pretrained: bool
        Whether to load pretrained weights or not

    model_args: dict
        Dictionary of model keyword arguments

    Returns
    -------
    model: torch.nn.Module
        Model
    """

    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        **model_args
    )
    # Override the head layer in transformer models so the output will be features
    model.head = nn.Identity()
    model.eval()

    return model


def prepare_dataloader(image_paths, transforms, batch_size, num_workers):

    """
    Prepare data loader for inference

    Parameters
    ----------
    image_paths: list of shape (n_images)
        List of image paths

    transforms: dict
        Transform pipeline

    batch_size: int
        Batch size of the data loader

    num_workers: int
        Number of workers of the data loader

    Returns
    -------
    data_loader: torch.utils.data.DataLoader
        Data loader
    """

    dataset = datasets.ImageDataset(image_paths=image_paths, transforms=transforms)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers
    )

    return data_loader


def extract_features(inputs, model, pooling_type, device, amp):

    """
    Extract features from given inputs with given model

    Parameters
    ----------
    inputs: torch.FloatTensor of shape (batch, channel, height, width)
        Inputs tensor

    model: torch.nn.Module
        Model

    pooling_type: str
        Pooling type applied to features

    device: torch.device
        Location of the inputs tensor and the model

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    features: torch.FloatTensor of shape (batch, features)
        Features tensor
    """

    with torch.no_grad():
        if amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                if pooling_type is not None:
                    # Use forward features if pooling type is specified (convolutional models)
                    features = model.forward_features(inputs)
                else:
                    # Use forward if pooling type is not specified (transformer models)
                    features = model(inputs)
        else:
            if pooling_type is not None:
                features = model.forward_features(inputs)
            else:
                features = model(inputs)

    features = features.detach().cpu()

    if pooling_type == 'avg':
        features = F.adaptive_avg_pool2d(features, output_size=(1, 1)).view(features.size(0), -1)
    elif pooling_type == 'max':
        features = F.adaptive_max_pool2d(features, output_size=(1, 1)).view(features.size(0), -1)
    elif pooling_type == 'concat':
        features = torch.cat([
            F.adaptive_avg_pool2d(features, output_size=(1, 1)).view(features.size(0), -1),
            F.adaptive_max_pool2d(features, output_size=(1, 1)).view(features.size(0), -1)
        ], dim=-1)

    features = features / features.norm(dim=1).view(-1, 1)

    return features
