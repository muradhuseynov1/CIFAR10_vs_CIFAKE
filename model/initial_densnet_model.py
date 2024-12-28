import torch.nn as nn
from torchvision import models

def create_densenet(num_classes=2):
    #pretrained densenet 121
    densenet = models.densenet121(pretrained=True)
    
    #changing last layers to fit our case - only 2 layers
    num_features = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
        nn.Softmax(dim=1)
    )
    return densenet