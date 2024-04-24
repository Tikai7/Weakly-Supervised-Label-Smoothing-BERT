import torch.nn as nn
import torch.nn.functional as F



class LSLoss(nn.Module):
    def __init__(self, smoothing, distribution) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.distribution = distribution

    def forward(self, y_pred, y_true):
        log_proba = F.log_softmax(y_pred)
        ground_truth_distribution = (1-self.smoothing)*y_true + self.smoothing*self.distribution
        

class T_LSLoss(nn.Module):
    pass

class WSLS(nn.Module):
    pass
