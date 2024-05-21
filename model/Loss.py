import torch
import torch.nn as nn
import torch.nn.functional as F

class LSmoothing(nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, targets, ns_scores=None, epsilon=0.1):
        # Create a tensor to hold the smoothed labels
        batch_size = targets.size(0)  # Number of examples in the batch
        smoothed_labels = torch.full((batch_size, self.num_classes), epsilon / (self.num_classes - 1), device=outputs.device)

        # Create a zero tensor the same size as smoothed_labels for one-hot encoding
        targets_one_hot = torch.zeros_like(smoothed_labels)  # Ensure device matches

        # Scatter 1s into the correct indices in targets_one_hot
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)  # No device argument here

        # Combine the uniform and one-hot distributions
        smoothed_labels = (1 - epsilon) * targets_one_hot + epsilon * smoothed_labels

        # Calculate the cross-entropy loss between the model's outputs and the smoothed labels
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(outputs, smoothed_labels)
    
    
class WSLSmoothing(nn.Module):
    """
    Integrates traditional label smoothing with Weakly Supervised Label Smoothing (WSLS) 
    as described in the article. This implementation is specifically designed for binary 
    classification where label 0 is considered the negative class.
    """
    def __init__(self, nb_labels=2, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ls = self.LSmoothing(nb_labels)
        
    def forward(self, y_pred, y_true, ns_scores, smoothing=0.1):
        """
        Args:
            y_pred: Predictions from the model (logits before softmax).
            y_true: Ground truth labels, expected to be binary (0 or 1).
            ns_scores: Precomputed negative sampling scores for each instance.
        """
        # Compute log probabilities from predictions
        log_probs = F.log_softmax(y_pred, dim=1)
        # The normal label smoothing for examples where label = 1 (i.e. the uniform distribution)
        _,y_0,y_1 = self.ls.forward(y_pred,y_true,smoothing)
        # Use weak supervision for labels = 0 
        # For the negative class (logits 0): Increase the smoothed value based on ns_scores
        weak_weight_neg = torch.where(y_true == 0, y_0 * ((1 - smoothing) + smoothing * ns_scores), y_0 * (smoothing + (1 - smoothing) * (1 - ns_scores)))
        # For the positive class (logits 1): Adjust by a small smoothing or use (1-ns_scores)
        weak_weight_pos = torch.where(y_true == 0,y_1 * (smoothing + (1 - smoothing) * ns_scores), y_1 * ((1 - smoothing) + smoothing * (1 - ns_scores)))
        # Create a tensor of these modified distributions
        weight = torch.stack([weak_weight_neg, weak_weight_pos], dim=1)
        # Compute the weighted loss
        loss = torch.sum(-weight * log_probs, dim=1).mean()
        return loss