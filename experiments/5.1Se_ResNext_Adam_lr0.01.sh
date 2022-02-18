### 5.1Se_ResNext_Adam_lr0.01
python main.py \
--lr 0.01 \
--data ./data \
--arch se_resnext_3474 \
--result Results_5.1Se_ResNext_Adam_lr0.01 \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--optimizer Adam \
--logfile 5.1Se_ResNext_Adam_lr0.01.txt \
--lr-type step



python utils/mail.py -a 5.1Se_ResNext_Adam_lr0.01