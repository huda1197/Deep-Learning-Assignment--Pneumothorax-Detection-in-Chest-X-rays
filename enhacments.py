import torch
import torch.nn as nn
import torch.nn.functional as F



class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Binary Cross-Entropy Loss
        bce_loss = self.bce_loss(inputs, targets)

        # Focal Loss
        probs = torch.sigmoid(inputs)
        focal_loss = self.alpha * (1 - probs) ** self.gamma * F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        focal_loss = focal_loss.mean()

        # Combine the losses
        combined_loss = bce_loss + focal_loss
        return combined_loss
    


