### 6.3Se_ResNext_WeightedCE_CBFinetune
python main.py \
--lr 0.001 \
--data ../../datasets/CV_project \
--arch se_resnext_3474 \
--result Results_6.3Se_ResNext_WeightedCE_CBFinetune \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--lr-type step \
--optimizer SGD \
--logfile 6.3Se_ResNext_WeightedCE_CBFinetune.txt \
--finetune True \
--weightloss True \
--epochs 40 \
--pth Results_6.1Se_ResNext_WeightedCE/se_resnext_3474_lr_0.1/model_best.pth.tar



python utils/mail.py -a 6.3Se_ResNext_WeightedCE_CBFinetune