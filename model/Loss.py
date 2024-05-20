import torch
import torch.nn as nn
import torch.nn.functional as F


class LSmoothing(nn.Module):
    def __init__(self, nb_labels=2) -> None:
        super().__init__()
        self.nb_labels = nb_labels
        self.uniform_distribution = 1 / nb_labels

    def forward(self, y_pred, y_true, smoothing=0.1):
        y_proba = F.log_softmax(y_pred, dim=1)
        smoothing_factor = 1 - smoothing
        uniform_smoothing_factor = smoothing / self.nb_labels

        smoothed_labels = torch.zeros_like(y_pred)
        for i in range(y_pred.size(0)):
            true_label = y_true[i]
            smoothed_labels[i] = uniform_smoothing_factor
            smoothed_labels[i, true_label] = smoothing_factor + uniform_smoothing_factor
            
        loss = -torch.sum(smoothed_labels * y_proba) / y_pred.size(0)
        return loss

