### 4.1.2Se_ResNext_warmup10cosine160
python main.py \
--lr 0.1 \
--data ../../datasets/CV_project \
--arch se_resnext_3474 \
--result Results_4.1.2Se_ResNext_warmup10cosine160 \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--optimizer SGD \
--logfile 4.1.2Se_ResNext_warmup10cosine160.txt \
--lr-type warmup_cos \
--warmup-epoch 10


python utils/mail.py -a 4.1.2Se_ResNext_warmup10cosine160