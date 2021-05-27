from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import torch
from utils.cutout import Cutout
from utils.transform import RandomRotate

normalize = transforms.Normalize(mean=[0.484, 0.460, 0.411],
                                 std=[0.260, 0.253, 0.271])

def transforms_train_val(crop_size, cutout_flag, RandomRotate_flag):
    """
        You can modify the train_transforms to try different image preprocessing methods when training model
    """
    if cutout_flag:
        if RandomRotate_flag:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(crop_size, padding=4),   #wht crop64
                transforms.RandomHorizontalFlip(),
                RandomRotate(15, 0.3),    # wht
                transforms.ToTensor(),
                normalize,
                Cutout(n_holes=2, length=8),    # cutout 2, 8
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(crop_size, padding=4),   #wht crop64
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                Cutout(n_holes=2, length=8),    # cutout 2, 8
            ])
    else:
        if RandomRotate_flag:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(crop_size, padding=4),   #wht crop64
                transforms.RandomHorizontalFlip(),
                RandomRotate(15, 0.3),    # wht
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(crop_size, padding=4),   #wht crop64
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, val_transforms

def transforms_test():
    """
        You can modify the function to try different image fusion methods when evaluating the trained model
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return trans
