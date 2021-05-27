### 3.3Se_ResNext_RandomRotate
python main.py \
--lr 0.1 \
--data ../../datasets/CV_project \
--arch se_resnext_3474 \
--result Results_3.3Se_ResNext_RandomRotate \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--lr-type step \
--optimizer SGD \
--logfile 3.3Se_ResNext_RandomRotate.txt \
--RandomRotate True


python utils/mail.py -a 3.3Se_ResNext_RandomRotate