import torch
import torch.nn as nn
class ClassificationLoss(nn.Module):
    def __init__(self):
        """Classification loss to DMFS."""
        super(ClassificationLoss, self).__init__()

    def forward(self, inputs, targets):
        loss_fn = nn.BCEWithLogitsLoss()
        output = loss_fn(inputs, targets)
        return output


