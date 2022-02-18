### 4.1.1Se_ResNext_warmup10step160
python main.py \
--lr 0.1 \
--data ./data \
--arch se_resnext_3474 \
--result Results_4.1.1Se_ResNext_warmup10step160 \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--optimizer SGD \
--logfile 4.1.1Se_ResNext_warmup10step160.txt \
--lr-type warmup \
--warmup-epoch 10


python utils/mail.py -a 4.1.1Se_ResNext_warmup10step160