### 6.1Se_ResNext_WeightedCE
python main.py \
--lr 0.1 \
--data # datapath \
--arch se_resnext_3474 \
--result Results_6.1Se_ResNext_WeightedCE \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--lr-type step \
--optimizer SGD \
--logfile 6.1Se_ResNext_WeightedCE.txt \
--weightloss True



python utils/mail.py -a 6.1Se_ResNext_WeightedCEs