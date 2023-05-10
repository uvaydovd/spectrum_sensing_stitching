import torch
from torch import nn

class BinaryFocalLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2, reduction=True) -> None:
        super(BinaryFocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction="none")
        self.reduction = reduction
    def forward(self,y_pred,y_true):
        if self.alpha is not None:
            alpha = torch.where(y_true == 1.0, self.alpha, (1.0 - self.alpha))
        else:
            alpha = 1
        pt = torch.where(y_true == 1.0, y_pred, 1 - y_pred)
        bce = self.bce(y_pred,y_true)
        loss = alpha*torch.pow(1.0 - pt, self.gamma)*bce
        if self.reduction:
            return torch.mean(loss)
        return torch.mean(loss,dim=(1,2))