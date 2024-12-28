import torch.nn as nn
from torchvision import models

def create_densenet(num_classes=2):
    #pretrained densenet 121
    densenet = models.densenet121(pretrained=True)
    
    #changing last layers to fit our case, changing 4 layers
    #batch normalisation added
    num_features = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128), 
        nn.Dropout(0.3),
        
        nn.Linear(128, 64), 
        nn.ReLU(),
        nn.BatchNorm1d(64),  
        nn.Dropout(0.3),

        nn.Linear(64, num_classes)  # Output layer
    )
    
    return densenet