import torch
import torch.nn.functional as F
import timm


def load_timm_model(model_name, pretrained, model_args):

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

    return model


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
                features = model.forward_features(inputs)
        else:
            features = model.forward_features(inputs)

    if pooling_type == 'avg':
        features = F.adaptive_avg_pool2d(features, output_size=(1, 1)).view(features.size(0), -1)
    elif pooling_type == 'max':
        features = F.adaptive_max_pool2d(features, output_size=(1, 1)).view(features.size(0), -1)
    elif pooling_type == 'concat':
        features = torch.cat([
            F.adaptive_avg_pool2d(features, output_size=(1, 1)).view(features.size(0), -1),
            F.adaptive_max_pool2d(features, output_size=(1, 1)).view(features.size(0), -1)
        ], dim=-1)

    features = features.detach().cpu()
    features = features / features.norm(dim=1).view(-1, 1).numpy()

    return features