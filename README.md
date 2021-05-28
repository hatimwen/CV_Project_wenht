# CV_Project_wenht

The codes of my CV Project.

My best result (0.67250) won the 7th place in the 'DLUT CV Project 2021: Image Classification' competition.

## Requirements

- python3.7

- numpy

- pytorch1.6

- torchvision

- pandas

## How to do the experiments I did?

- **[Optionally]** Change line 15~21 in [utils/mail.py](utils/mail.py#L15) according to your information.
**Note** that you should comment the line with 'mail' if you don't want to receive the e-mail when your training-process ends.

- Change data_path in `main.py` and others.

- And then, if you want to run **2.1crop64_Baseline_step160**, just

```shell
sh experiments/2.1crop64_Baseline_step160.sh
```

- You can download my best checkpoints `0524_kdMSE_finetune250_pre_Results_step50.pth.tar` with many tricks [here](http://pan.dlut.edu.cn/share?id=i19jtjtddsyg).

## Importantly! How to get my best score?

### Finetune resnet_t

1. train resnet_t (resnet-101) with [official pretrained weight](http://pan.dlut.edu.cn/share?id=i1m3d8tddnu6), remenber to **freeze** the layers before `fc` layers with:

```python
# change L130~173:
with torch.no_grad():
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
x = x.view(x.size(0), -1)
x = self.fc(x)
```

Then, you will get the weight of the fc-finetuned model.

You can also download `resnet101_fine_fc.pth.tar` [here](http://pan.dlut.edu.cn/share?id=idg771td9e3w) and put it in `./pretrained`.

2. train resnet_t (resnet-101) with the pretrained weight you gain below, remenber **not to freeze anything** to finetune the whole model this time.

```python
# change L130~173:
x = self.conv1(x)
x = self.bn1(x)
x = self.relu(x)

x = self.layer1(x)
x = self.layer2(x)
x = self.layer3(x)
x = self.layer4(x)

x = self.avgpool(x)
x = x.view(x.size(0), -1)
x = self.fc(x)
```

Then, you will get the weight of the whole finetuned model.

You can also download `resnet101_fine_all.pth.tar` [here](http://pan.dlut.edu.cn/share?id=idnd3mtd99y3) and put it in `./pretrained`.

### Train my se_resnext_3474

For convenience， I put all the programs you should run in [all_in.sh](experiments/all_in.sh), so just:

```shell
sh experiments/all_in.sh
```

**Note** that some paths should be changed.

## Some Experiments

### 2 Network(crop64)

#### 2.1crop64_Baseline_step160

#### 2.2Se_ResNext_step160

### 3 Data augmentation

#### 3.1Se_ResNext_CutOut

#### 3.2Se_ResNext_MixUp

#### 3.3Se_ResNext_RandomRotate

### 4 Training strategy

#### 4.1.1Se_ResNext_warmup10step160

#### 4.1.2Se_ResNext_warmup10cosine160

#### 4.2Se_ResNext_step160_labelsmooth

### 5 Optimization

#### 5.1Se_ResNext_Adam_lr0.001

#### 5.2Se_ResNext_RMSprop_lr0.001

### 6 Long-tail distribution

#### 6.1Se_ResNext_WeightedCE

#### 6.2Se_ResNext_CBFinetune

#### 6.3Se_ResNext_WeightedCE_CBFinetune

## Contact

- Author: Hatimwen

- Email: hatimwen@163.com
