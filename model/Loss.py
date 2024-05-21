import torch
import torch.nn as nn
import torch.nn.functional as F

class LSmoothing(nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, outputs, targets, ns_scores=None, epsilon=0.1, return_sl=False):
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
        if return_sl:
            return smoothed_labels
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
        self.ls = LSmoothing(nb_labels)
        self.num_classes = 2
        

    def forward(self, outputs, targets, ns_scores, smoothing=0.1):
        ns_scores = ns_scores.unsqueeze(-1)  # Ensure ns_scores is always [N, 1]

        # Compute smoothed labels using the original label smoothing class
        smoothed_labels = self.ls(outputs, targets, ns_scores, smoothing, True)
        adjusted_labels = torch.zeros_like(smoothed_labels)

        # Indices for positive and negative logits
        pos_indices = targets == 1
        neg_indices = targets == 0

        # Adjust the positive and negative class weights using ns_scores
        adjusted_labels[pos_indices, 1] = (1 - smoothing) + smoothing * ns_scores[pos_indices].squeeze(-1)
        adjusted_labels[neg_indices, 0] = (1 - smoothing) * ns_scores[neg_indices].squeeze(-1) + smoothing / (self.num_classes - 1)
        adjusted_labels[neg_indices, 1] = (1 - smoothing) * (1 - ns_scores[neg_indices].squeeze(-1)) + smoothing / (self.num_classes - 1)

        # Calculate the cross-entropy loss using the adjusted labels
             
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(outputs, adjusted_labels)