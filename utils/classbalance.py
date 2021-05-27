import os, random, glob
from shutil import copyfile, move
from tqdm import tqdm

def moveFile(inputDir, outDir, count, choseImg):
    #通过glob.glob来获取原始路径下，所有'.jpg'文件
    imageList1 = glob.glob(os.path.join(inputDir, '*.png'))
    #随机获取count数量的jpg文件
    imageRandom = random.sample(imageList1, count)

    #遍历所有随机得到的jpg文件，获取文件名称（包括后缀）
    for item in imageRandom:
        choseImg.append(os.path.basename(item))

    #os.path.splitext()，返回元组，为文件名称与文件后缀格式
    for item in choseImg:
        #将随机选中的jpg文件遍历复制到目标文件夹中
        inputDir1 = os.path.abspath(inputDir+'/'+item)
        outDir1 = os.path.abspath(outDir+'/'+item)
        mkdir(outDir)
        move(inputDir1, outDir1)

def copyFile(inputDir, outDir, count, choseImg):
    #通过glob.glob来获取原始路径下，所有'.jpg'文件
    imageList1 = glob.glob(os.path.join(inputDir, '*.png'))
    #随机获取count数量的jpg文件
    imageRandom = random.sample(imageList1, count)

    #遍历所有随机得到的jpg文件，获取文件名称（包括后缀）
    for item in imageRandom:
        choseImg.append(os.path.basename(item))

    #os.path.splitext()，返回元组，为文件名称与文件后缀格式
    for item in choseImg:
        #将随机选中的jpg文件遍历复制到目标文件夹中
        inputDir1 = os.path.abspath(inputDir+'/'+item)
        outDir1 = os.path.abspath(outDir+'/'+item)
        mkdir(outDir)
        copyfile(inputDir1, outDir1)
    return

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print
        "---  new folder...  ---"
        print
        "---  OK  ---"

    else:
        print
        "---  There is this folder!  ---"

def classbalance(num_class=200, cp_num=50, datapath='./data'):
    for i in tqdm(range(num_class),desc='Create train_{}'.format(cp_num)):
        choseImg = []
        inputfile = os.path.join(datapath, 'train/' + str(i).zfill(4))
        outfile = os.path.join(datapath, 'train_{}/'.format(cp_num) + str(i).zfill(4))
        copyFile(inputfile, outfile, cp_num, choseImg)
        # moveFile(inputfile, outfile, cp_num, choseImg)

# if __name__ == '__main__':
#     all_num = 200
#     count = 100
#     for i in tqdm(range(all_num)):
#         choseImg = []
#         inputfile = '../../datasets/CV_project/val_100/' + str(i).zfill(4)
#         outfile = '../../datasets/CV_project/train_150/' + str(i).zfill(4)
#         #指定找到文件后，另存为的文件夹路径
#         # outDir = os.path.abspath(outfile)
#         #指定文件的原始路径
#         # inputDir = os.path.abspath(inputfile)
#         # copyFile(inputfile, outfile, count, choseImg)
#         moveFile(inputfile, outfile, count, choseImg)