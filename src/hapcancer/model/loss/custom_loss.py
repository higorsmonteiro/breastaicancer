'''

'''
import torch.nn as nn
import torch.nn.functional as F
from libauc.losses import AUCMLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        """
            Focal Loss for imbalanced classification.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets):
        """
            Args:
            -----
                preds: model outputs (probabilities) (batch_size, 1)
                targets: labels (batch_size, 1)
        """
        bce_loss = F.binary_cross_entropy(preds, targets, reduction="none")
        p_t = preds * targets + (1 - preds) * (1 - targets) 
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()

# --- helper: masked interval BCE over logits (shape BxT) ---
def interval_bce_loss(logits_h, y, mask, pos_weight=None):
    """
    logits_h: (B,T) interval hazard logits for each year-interval
    y       : (B,T) interval labels (0/1) - first-event-in-interval targets
    mask    : (B,T) 1 where label is observed (include in loss), 0 otherwise
    pos_weight: optional tensor/list of length T with per-interval pos_weight
    """
    assert logits_h.ndim == y.ndim == mask.ndim == 2, "Expect (B,T) tensors"
    B, T = logits_h.shape
    loss_sum, denom = 0.0, 0.0
    for t in range(T):
        logit_t = logits_h[:, t]
        y_t     = y[:, t].float()
        m_t     = mask[:, t].float()
        if pos_weight is None:
            l_t = F.binary_cross_entropy_with_logits(logit_t, y_t, reduction='none')
        else:
            # pos_weight can be a scalar or 1D tensor length T
            pw = pos_weight[t] if isinstance(pos_weight, (list, tuple, torch.Tensor)) else pos_weight
            l_t = F.binary_cross_entropy_with_logits(logit_t, y_t, reduction='none', pos_weight=pw)
        obs = m_t.sum().clamp_min(1.0)
        loss_sum += (l_t * m_t).sum() / obs
        denom += 1.0
    return loss_sum / denom


def build_interval_aucm_losses(T, imratio_per_interval=None, device='cuda'):
    """Return a list of LibAUC AUCMLoss objects, one per interval t."""
    losses = []
    for t in range(T):
        imr = None if imratio_per_interval is None else float(imratio_per_interval[t])
        # You can tune margin; 1.0 is a common default
        losses.append(AUCMLoss(margin=1.0, imratio=imr).to(device))
    return losses

def interval_aucm_loss(logits_h, y, mask, aucm_losses):
    """
    logits_h, y, mask: (B,T)
    aucm_losses: list of length T with AUCMLoss instances
    Returns mean over intervals observed.
    """
    B, T = logits_h.shape
    total, cnt = 0.0, 0
    for t in range(T):
        m_t = mask[:, t].bool()
        if m_t.any():
            # LibAUC expects 1D tensors of scores and labels
            scores_t = logits_h[m_t, t]
            labels_t = y[m_t, t].float()
            total += aucm_losses[t](scores_t, labels_t)
            cnt += 1
    return total / max(1, cnt)
        