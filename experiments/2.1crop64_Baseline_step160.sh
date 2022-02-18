### 2.1crop64_Baseline_step160
python main.py \
--lr 0.1 \
--data \
--arch resnet35 \
--result Results_2.1crop64_Baseline_step160 \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--lr-type step \
--optimizer SGD \
--logfile 2.1crop64_Baseline_step160.txt

python utils/mail.py -a 2.1crop64_Baseline_step160
