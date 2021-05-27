### 2.2Se_ResNext_step160
python main.py \
--lr 0.1 \
--data ../../datasets/CV_project \
--arch se_resnext_3474 \
--result Results_2.2Se_ResNext_step160 \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--lr-type step \
--optimizer SGD \
--logfile 2.2Se_ResNext_step160.txt

python utils/mail.py -a 2.2Se_ResNext_step160