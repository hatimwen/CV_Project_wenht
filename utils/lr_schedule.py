import math
def adjust_learning_rate(args, optimizer, epoch):
    """
     For AlexNet, the lr starts from 0.05, and is divided by 10 at 90 and 120 epochs
    """
    if args.lr_type == 'step':
        if epoch < 0.5625*args.epochs:
            lr = args.lr
        elif epoch < 0.75*args.epochs:
            if args.finetune:
                lr = args.lr * 0.2 #wht finetune150
            else:
                lr = args.lr * 0.1
        else:
            if args.finetune:
                lr = args.lr * 0.04 #wht finetune150
            else:
                lr = args.lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.lr_type == 'warmup':
        if epoch < args.warmup_epoch:
            lr = (epoch+1) / args.warmup_epoch * args.lr
        elif epoch < 0.5625*args.epochs:
            lr = args.lr
        elif epoch < 0.75*args.epochs:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.lr_type == 'warmup_cos':
        if epoch < args.warmup_epoch:
            lr = (epoch+1) / args.warmup_epoch * args.lr
        else:
            # import math
            lr = 0.5 * (math.cos((epoch - args.warmup_epoch) /(args.epochs - args.warmup_epoch) * math.pi) + 1) * args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.lr_type == 'Tcos':
        if epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        elif (epoch - 1 - args.T_max) % (2 * args.T_max) == 0:
            # import math
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] + args.lr * \
                        (1 - math.cos(math.pi / args.T_max)) / 2
        else:
            # import math
            for param_group in optimizer.param_groups:
                param_group['lr'] *= (1 + math.cos(math.pi * epoch / args.T_max)) / \
                                    (1 + math.cos(math.pi * (epoch - 1) / args.T_max))
        
        # import torch
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)
        # 0.5 * (math.cos((epoch - args.warmup_epoch) /(args.epochs - args.warmup_epoch) * math.pi) + 1) * args.lr
    else:
        raise KeyError('learning_rate schedule method {0} is not achieved'.format(args.lr_type))

