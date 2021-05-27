import numpy as np
import torch
import torch.nn as nn

def mixup(batch_x, batch_y, alpha=0.5, use_cuda=True):
    '''
    batch_x：批样本数，shape=[batch_size,channels,width,height]
    batch_y：批样本标签，shape=[batch_size]
    alpha：生成lam的beta分布参数，一般取0.5效果较好
    use_cuda：是否使用cuda
    
    returns：
        mixed inputs, pairs of targets, and lam
    '''
    
    if alpha > 0:
        #alpha=0.5使得lam有较大概率取0或1附近
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = batch_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size) #生成打乱的batch_size索引
        
    #获得混合的mixed_batchx数据，可以是同类（同张图片）混合，也可以是异类（不同图片）混合
    mixed_batchx = lam * batch_x + (1 - lam) * batch_x[index, :]
    
    """
    Example：
    假设batch_x.shape=[2,3,112,112]，batch_size=2时，
    如果index=[0,1]的话，则可看成mixed_batchx=lam*[[0,1],3,112,112]+(1-lam)*[[0,1],3,112,112]=[[0,1],3,112,112]，即为同类混合
    如果index=[1,0]的话，则可看成mixed_batchx=lam*[[0,1],3,112,112]+(1-lam)*[[1,0],3,112,112]=[batch_size,3,112,112]，即为异类混合
    """
    batch_ya, batch_yb = batch_y, batch_y[index]
    return mixed_batchx, batch_ya, batch_yb, lam