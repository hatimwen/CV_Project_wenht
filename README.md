# CV_Project_wht

The codes of my CV Project.

My best result (0.67250) won the 7th place in the 'DLUT CV Project 2021: Image Classification' competition.

## Requirements

- python3.7

- numpy

- pytorch1.6

- torchvision

- pandas

## How to use

- **[Optionally]** Change line 15~21 in [utils/mail.py](utils/mail.py#L15) according to your information.
**Note** that you should comment the line with 'mail' if you don't want to receive the e-mail when your training-process ends.

- Change data_path in `main.py` and others.

- And then, if you want to run **2.1crop64_Baseline_step160**, just

```shell
sh experiments/2.1crop64_Baseline_step160.sh
```

- You can download my best checkpoints with many tricks [here](http://pan.dlut.edu.cn/share?id=ig2fuktdcy3s)

## Experiments

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
