### 5.1Se_ResNext_Adam_lr0.001
python main.py \
--lr 0.001 \
--data \
--arch se_resnext_3474 \
--result Results_5.1Se_ResNext_Adam_lr0.001 \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--optimizer Adam \
--logfile 5.1Se_ResNext_Adam_lr0.001.txt \
--lr-type step



python utils/mail.py -a 5.1Se_ResNext_Adam_lr0.001