### 3.1Se_ResNext_CutOut
python main.py \
--lr 0.1 \
--data ./data \
--arch se_resnext_3474 \
--result Results_3.1Se_ResNext_CutOut \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--lr-type step \
--optimizer SGD \
--logfile 3.1Se_ResNext_CutOut.txt \
--cutout True

python utils/mail.py -a 3.1Se_ResNext_CutOut