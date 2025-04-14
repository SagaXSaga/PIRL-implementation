import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

class PIRL(nn.Module):
    """
    PIRL class with Model and Loss components.
    """
    def __init__(self, encoding_size=128, pretrained=True, loss_lambda=0.8):
        """
        Initialize the PIRL model with both the Model and Loss submodules.
        """
        super(PIRL, self).__init__()
        self.model = self.Model(encoding_size=encoding_size, pretrained=pretrained)
        self.loss = self.Loss(loss_lambda=loss_lambda)

    def forward(self, x, transformed_x=None):
        """
        If only x is provided, return its feature encoding.
        If transformed_x is provided (e.g., for a pretext task like jigsaw), return
        a tuple: (encoding for x, encoding for transformed_x).
        """
        output = self.model(x)
        if transformed_x is not None:
            transformed_output = self.model(transformed_x)
            return output, transformed_output
        return output

    class Model(nn.Module):
        """
        The model builds feature representations using ResNet50
        and a projection head to map features to a lower encoding space.
        """
        def __init__(self, encoding_size=128, pretrained=True):
            super(PIRL.Model, self).__init__()
            backbone = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-1])  # Remove final FC
            self.projection = nn.Linear(2048, encoding_size)  # Projection head
            self.classifier = nn.Linear(encoding_size, 10)    # For CIFAR-10

        def forward(self, x):
            # Extract features from the backbone.
            x = self.features(x)       # [B, 2048, 1, 1]
            x = torch.flatten(x, 1)    # [B, 2048]
            features = self.projection(x)  # [B, encoding_size]
            logits = self.classifier(features)  # [B, 10]
            return logits, features

    class Loss(nn.Module):
        """
        The loss module computes a weighted sum of the cross-entropy losses
        from two representations: the original and the transformed version.
        """
        def __init__(self, loss_lambda=0.5):
            super(PIRL.Loss, self).__init__()
            self.loss_lambda = loss_lambda

        def forward(self, output, transformed_output, target):
            """
            Computes the loss
            output: predictions from the original image.
            transformed_output: predictions from the transformed image.
            target: ground-truth labels
            """
            loss1 = F.cross_entropy(output, target)
            loss2 = F.cross_entropy(transformed_output, target)
            total_loss = self.loss_lambda * loss1 + (1 - self.loss_lambda) * loss2
            return total_loss



class AverageMeter:
    """Computes and stores the average, current value, sum, and count. From keras"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count