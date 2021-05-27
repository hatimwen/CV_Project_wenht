import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import argparse

def get_num_list(dataroot):
    # Return:
    # num_class_list: Num of each cat, int numpy array, [50, 900]
    # num_classes: cat list , str numpy array, ['0000'~'0199']
    train_root = osp.join(dataroot, 'train')
    num_classes = np.sort(os.listdir(train_root))
    num_class_list = np.zeros(len(num_classes), dtype=np.int)
    for i, cat in tqdm(enumerate(num_classes)):
        num_class_list[i] += len(os.listdir(osp.join(train_root, cat)))
    num_classes = np.array(num_classes)
    num_class_list = np.array(num_class_list)
    return num_class_list, num_classes

def get_cat_num(cat, num_class_list, num_classes):
    return num_class_list[np.where(num_classes == cat)][0]

def get_weight_list(num_class_list):
    return np.array([min(num_class_list) / N for N in num_class_list])


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='PyTorch ImageNet100_64*64 Training')
#     parser.add_argument('--data', default='./data', type=str, metavar='N',
#                         help='root directory of dataset where directory train_data or val_data exists')
#     args = parser.parse_args()
#     num_class_list, num_classes = get_num_list(args.data)
#     cat = '0199'
#     num = get_cat_num(cat, num_class_list, num_classes)
#     print('Num of cat \'{}\' is {}.'.format(cat, num))
#     weight = get_weight_list(num_class_list)
#     print('*'*10, 'num_class_list','*'*10)
#     print(num_class_list)
#     print('*'*10, 'weight','*'*10)
#     print(weight)
#     print('*'*10, 'weight.min','*'*10)
#     print(weight.min())
#     print('*'*10, 'weight.max','*'*10)
#     print(weight.max())
#     print('*'*10)
