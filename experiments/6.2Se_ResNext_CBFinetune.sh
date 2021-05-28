### 6.2Se_ResNext_CBFinetune
python main.py \
--lr 0.001 \
--data # datapath \
--arch se_resnext_3474 \
--result Results_6.2Se_ResNext_CBFinetune \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--lr-type step \
--optimizer SGD \
--logfile 6.2Se_ResNext_CBFinetune.txt \
--finetune True \
--epochs 40 \
--pth Results_2.2Se_ResNext_step160/se_resnext_3474_lr_0.1/model_best.pth.tar



python utils/mail.py -a 6.2Se_ResNext_CBFinetune