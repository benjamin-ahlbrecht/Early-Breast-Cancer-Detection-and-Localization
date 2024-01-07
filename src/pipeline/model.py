import torch
from torch import nn
from torchvision.models import efficientnet_b4


def custom_efficientnet_b4(device: torch.device, progress: bool = True):
    classifier = efficientnet_b4(weights="IMAGENET1K_V1", progress=True)
    classifier = classifier.to(device)

    # Freeze all layers
    for parameter in classifier.parameters():
        parameter.requires_grad = False
        
    # Unfreeze the desired last blocks for training
    gradient_blocks = [7, 8]
    for block in gradient_blocks:
        for parameter in classifier.features[block].parameters():
            parameter.requires_grad = True

    # Replace the last classification layer with a single node for Positive/Negative
    classifier.classifier[1] = torch.nn.Linear(1792, 1)

    # Unfreeze the classification layer as well...
    for parameter in classifier.classifier.parameters():
        parameter.requires_grad = True

    # Add a sigmoid activation function at the end of the classification layer
    classifier.classifier.append(nn.Sigmoid())

    return classifier