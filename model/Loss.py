import torch
import torch.nn as nn
import torch.nn.functional as F

class LSmoothing(nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, outputs, targets, ns_scores = None, epsilon=0.1, return_sl=False):
        batch_size = targets.size(0)
        smoothed_labels = torch.full((batch_size, self.num_classes), epsilon / (self.num_classes - 1), device=outputs.device)
        targets_one_hot = torch.zeros_like(smoothed_labels)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        smoothed_labels = (1 - epsilon) * targets_one_hot + epsilon * smoothed_labels

        # A utlilisé pour WSLS
        if return_sl:
            return smoothed_labels
        
        # Calculer la cross-entropy avec les "smoothed labels"
        log_probs = F.log_softmax(outputs, dim=1)
        loss_fn = nn.KLDivLoss(reduction='batchmean')
        return loss_fn(log_probs, smoothed_labels)
    
    
class WSLSmoothing(nn.Module):

    def __init__(self, nb_labels=2, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ls = LSmoothing(nb_labels)
        self.num_classes = 2
        
    def forward(self, outputs, targets, ns_scores, smoothing=0.1):
        ns_scores = ns_scores.unsqueeze(-1)  # Ensure ns_scores is always [N, 1]

        # Récuperé les smoothed_labels
        smoothed_labels = self.ls(outputs, targets, epsilon=smoothing, return_sl=True)
        adjusted_labels = smoothed_labels.clone() 

        # Indices des classe negative
        neg_indices = targets == 0

        # Ajusté les poids des classes négatives avec les scores de NS (formule du papier)
        adjusted_labels[neg_indices, 0] = (1 - smoothing) * ns_scores[neg_indices].squeeze(-1) + smoothing / (self.num_classes - 1)
        adjusted_labels[neg_indices, 1] = (1 - smoothing) * (1 - ns_scores[neg_indices].squeeze(-1)) + smoothing / (self.num_classes - 1)

        # Calculer la cross-entropy avec les "adjusted labels"
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(outputs, adjusted_labels)