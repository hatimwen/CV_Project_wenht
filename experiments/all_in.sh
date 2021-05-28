## my best plan here:

### train se_resnext_3474 with many many tricks first:
python main.py \
--lr 0.1 \
--data # datapath \
--arch se_resnext_3474 \
--result ./Results_final \
--crop-size 64 \
--loss-type labelsmooth \
--optimizer SGD \
--logfile all_in.txt \
--epochs 300 \
--lr-type warmup \
--warmup-epoch 20 \
--mixup True \
--cutout True

### then, train(finetune) se_resnext_3474 with kd:
python main_kd.py \
--data # datapath \
--arch se_resnext_3474 \
--kdarch resnet101_t \
--result ./Results_final_kd \
--logfile all_in_kd.txt \
--epochs 80 \
--lr-type step \
--lr 0.001 \
--mixup True \
--cutout True \
--pretrained True \
--pth Results_final/se_resnext_3474_lr_0.1/model_best.pth.tar \
--tpth pretrained/resnet101_fine_all.pth.tar

### lastly, finetune with ClassBalance:
python main.py \
--lr 0.001 \
--data # datapath \
--arch se_resnext_3474 \
--result ./Results_final_finetune \
--crop-size 64 \
--loss-type CrossEntropyLoss \
--lr-type step \
--optimizer SGD \
--logfile final_finetune.txt \
--finetune True \
--epochs 40 \
--pth Results_final_kd/se_resnext_3474_lr_0.001/model_best.pth.tar
