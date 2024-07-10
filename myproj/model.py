import torch.nn as nn
import torchvision.models as models


def get_model(config):
    model = models.resnet18(pretrained=config.model.pretrained)
    # Modify the final layer based on the number of classes (if necessary)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.dataset.num_classes)
    return model
