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


class WSLSmoothing(nn.Module):
    """
    Integrates traditional label smoothing with Weakly Supervised Label Smoothing (WSLS) 
    as described in the article. This implementation is specifically designed for binary 
    classification where label 0 is considered the negative class.
    """
    def __init__(self, nb_labels=2, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ls = LSmoothing(nb_labels)
        
    def forward(self, y_pred, y_true, ns_scores):
        """
        Args:
            y_pred: Predictions from the model (logits before softmax).
            y_true: Ground truth labels, expected to be binary (0 or 1).
            ns_scores: Precomputed negative sampling scores for each instance.
        """
        # Compute log probabilities from predictions
        log_probs = F.log_softmax(y_pred, dim=1)
        # The normal label smoothing for examples where label = 1 (i.e. the uniform distribution)
        _,y_0,y_1 = self.ls.forward(y_pred,y_true,self.smoothing)
        
        # Use weak supervision for labels = 0 
        # For the negative class (logits 0): Increase the smoothed value based on ns_scores
        weak_weight_neg = torch.where(y_true == 0, y_0 * ((1 - self.smoothing) + self.smoothing * ns_scores), y_0 * (self.smoothing + (1 - self.smoothing) * (1 - ns_scores)))
        # For the positive class (logits 1): Adjust by a small smoothing or use (1-ns_scores)
        weak_weight_pos = torch.where(y_true == 0,y_1 * (self.smoothing + (1 - self.smoothing) * ns_scores),y_1 * ((1 - self.smoothing) + self.smoothing * (1 - ns_scores)))

        # Create a tensor of these modified distributions
        weight = torch.stack([weak_weight_neg, weak_weight_pos], dim=1)
        
        # Compute the weighted loss
        loss = torch.sum(-weight * log_probs, dim=1).mean()

        return loss

