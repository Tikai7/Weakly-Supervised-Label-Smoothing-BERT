import torch.nn as nn
import torch.nn.functional as F


class LSmoothing(nn.Module):
    def __init__(self, nb_labels=2) -> None:
        super().__init__()
        self.uniform_distribution = 1/nb_labels

    def forward(self, y_pred, y_true, smoothing=0.1):
        y_proba = F.log_softmax(y_pred, dim=1)
        y_proba_inv = F.log_softmax((1-y_pred), dim=1)

        proba_c0, proba_c1 = y_proba[0]
        proba_c0_inv, proba_c1_inv = y_proba_inv[0]

        groung_truth_c0, groung_truth_c1 = y_true[0]

        smoothing_factor = (1-smoothing)
        uniform_smoothing_factor = smoothing*self.uniform_distribution

        y_0 = (smoothing_factor*groung_truth_c0 + uniform_smoothing_factor)
        y_1 = (smoothing_factor*groung_truth_c1 + uniform_smoothing_factor)
        
        loss = -self.uniform_distribution*(y_0 * proba_c0 + (1-y_0)*proba_c0_inv + y_1*proba_c1 + (1-y_1)*proba_c1_inv)

        return loss, y_0, y_1




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