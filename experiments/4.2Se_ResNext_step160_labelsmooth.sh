### 4.2Se_ResNext_step160_labelsmooth
python main.py \
--lr 0.1 \
--data \
--arch se_resnext_3474 \
--result Results_4.2Se_ResNext_step160_labelsmooth \
--crop-size 64 \
--optimizer SGD \
--logfile 4.2Se_ResNext_step160_labelsmooth.txt \
--lr-type step \
--loss-type labelsmooth



python utils/mail.py -a 4.2Se_ResNext_step160_labelsmooth