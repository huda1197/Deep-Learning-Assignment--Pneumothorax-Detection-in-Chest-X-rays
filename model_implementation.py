import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModifiedResNet50FT(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedResNet50FT, self).__init__()
        # Load the pretrained ResNet-50 model
        self.model = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer to accept single channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=(2,2), padding=(3,3), bias=False)
        
        # Modify the fully connected layer to output a single value for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedResNet50, self).__init__()
        # Load the pretrained ResNet-50 model
        self.model = models.resnet50(pretrained=True)
    
        # Modify the first convolutional layer to accept single channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

                # Remove the fully connected layer (we will replace it)
        self.model.fc = nn.Identity() 
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling (output size: 1x1)
        self.fc = nn.Linear(2048, num_classes)

    
    def forward(self, x):
    # Forward pass through the ResNet-50 layers (without the final FC layer)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
    
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
    
    # Global Average Pooling
        x = self.gap(x)  # Output shape: (batch_size, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten the tensor to shape (batch_size, 2048)
    
    # Fully connected layer for binary classification
        x = self.fc(x)  # Output shape: (batch_size, num_classes)
    
        return x