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

        return loss
class WSLSmoothing(nn.Module):
    pass
