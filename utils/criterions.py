import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, logits, targets, weights=None):
        if weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
        weights.requires_grad = False
        
        C = logits.size(1)
        if logits.dim() == 5:
            logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, C)   # [B*H*W, C]
        else:
            logits = logits.permute(0, 2, 3, 1).reshape(-1, C)   # [B*H*W, C]
            
        targets = targets.view(-1)                          # [B*H*W]
        weights = weights.view(-1)
        
        loss = F.cross_entropy(logits, targets.long(), reduction='none', ignore_index=-1)
        loss = loss * weights
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def dice_coeff(self, score, target, weights):
        target = target.float()
        intersect = torch.sum(score * target * weights)
        y_sum = torch.sum(target * target * weights)
        z_sum = torch.sum(score * score * weights)
        loss = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        return loss
        
    def forward(self, score, target, weights):
        weights.requires_grad = False
        loss_val = 0.
        score_softmax = torch.softmax(score, dim=1)
        n_classes = score.size(1)
        for i in range(1, n_classes):
            loss_val += (1 - self.dice_coeff(score_softmax[:, i], target==i, weights=weights))
             
        return loss_val / n_classes
    
class CompoundLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = CELoss()
        
    def forward(self, logits, targets, weights):
        l_dice = self.dice(logits, targets, weights)
        l_ce = self.ce(logits, targets, weights)
        
        return (l_dice + l_ce) * 0.5
        

class SoftmaxMSE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, targets):
        
        logits = torch.softmax(logits, dim=1)
        targets = torch.softmax(targets, dim=1)
        
        return F.mse_loss(logits, targets.detach())
