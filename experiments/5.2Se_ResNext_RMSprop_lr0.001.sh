### 5.2Se_ResNext_RMSprop_lr0.001
python main.py \
--lr 0.001 \
--data ../../datasets/CV_project \
--arch se_resnext_3474 \
--result Results_5.2Se_ResNext_RMSprop_lr0.001 \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--optimizer RMSprop \
--logfile 5.2Se_ResNext_RMSprop_lr0.001.txt \
--lr-type step



python utils/mail.py -a 5.2Se_ResNext_RMSprop_lr0.001