import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch import Tensor
from typing import Optional

class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()
        self.para_dict = para_dict
        self.device = torch.device("cuda")
        if para_dict == None:
            self.weight_list = None
        else:
            self.num_classes = self.para_dict["num_classes"]
            self.num_class_list = self.para_dict['num_class_list']
            self.weight_list = torch.FloatTensor(np.array([min(self.num_class_list) / N for N in self.num_class_list])).to(self.device)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, weight=self.weight_list)

class FocalLoss(nn.Module):
    """
        Equation: Loss(x, class) = - (1-sigmoid(p^t))^gamma \log(p^t)
        
        Focal loss tries to make neural networks to pay more attentions on difficult samples.
    """
    def __init__(self, class_num=200, gamma=2, alpha=None, reduction='mean', para_dict=None):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).cuda()
        else:
            self.alpha = alpha.cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, 200) #获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor
#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print((torch.pow((1 - probs), self.gamma)))
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class CrossEntropyLabelSmooth(CrossEntropy):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

        Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight of label smooth.
    """
    def __init__(self, para_dict=None, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__(para_dict)
        self.epsilon = epsilon #hyper-parameter in label smooth
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.num_classes = self.para_dict["num_classes"]

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class KDLoss(nn.Module):
    def __init__(self, para_dict=None):
        super(KDLoss, self).__init__()
        self.para_dict = para_dict
        self.device = torch.device("cuda")
        self.num_class_list = para_dict['num_class_list']
        self.weight_list = torch.FloatTensor(np.array([min(self.num_class_list) / N for N in self.num_class_list])).to(self.device)
        self.alpha = para_dict['alpha']
        self.T = para_dict['temperature']
    def forward(self, outputs, labels, teacher_outputs):
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/self.T, dim=1), \
                                F.softmax(teacher_outputs/self.T, dim=1)) * (self.alpha * self.T * self.T) + \
                                F.cross_entropy(outputs, labels, self.weight_list) * (1. - self.alpha)
        return KD_loss
