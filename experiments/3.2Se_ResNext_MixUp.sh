### 3.2Se_ResNext_MixUp
python main.py \
--lr 0.1 \
--data ../../datasets/CV_project \
--arch se_resnext_3474 \
--result Results_3.2Se_ResNext_MixUp \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--lr-type step \
--optimizer SGD \
--logfile 3.2Se_ResNext_MixUp.txt \
--mixup True


python utils/mail.py -a 3.2Se_ResNext_MixUp