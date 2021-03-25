import os.path as osp

import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader.samplers import FewShotSampler
from util.utils import set_gpu, Averager, Timer, set_seed
from util.metric import compute_confidence_interval, count_acc
from util.args_parser import get_args, process_args, print_args, init_saving_features, init_saving_icn_scores
from util.logger import ExpLogger


if __name__ == '__main__':
    
    print('###### Start experiment with args: ######')
    args = get_args()
    process_args(args)
    print_args(args)

    set_seed(args.seed)
    set_gpu(args.gpu)
    cudnn.benchmark = True

    timer = Timer()
    timer.start()

    explog = ExpLogger(args)
    writer = SummaryWriter(log_dir=args.save_path)

    print('###### Load data ######')
    if args.dataset == 'MiniImageNet':
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from dataloader.tiered_imagenet import tieredImageNet as Dataset       
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args)
    train_sampler = FewShotSampler(trainset.label, args.train_epi, args.train_way, args.shot, args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = FewShotSampler(valset.label, args.val_epi, args.way, args.shot, args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)
    

    print('###### Create model ######')
    if args.model == 'ProtoNet':
        from models.protonet import ProtoNet as Model
    else:
        raise ValueError('Non-supported Model.')
    model = Model(args)
    
    # load pre-trained model (no FC weights)
    model_dict = model.state_dict()
    if args.init_weights is not None:
        model_detail = torch.load(args.init_weights)
        if 'params' in model_detail:
            pretrained_dict = model_detail['params']
            # remove weights for FC
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print('Pretrained dict keys:')
            print(pretrained_dict.keys())
        else:
            pretrained_dict = model_detail['model']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
        model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)    
    model = model.cuda()
    # model = torch.nn.DataParallel(model.cuda())

    explog.parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('###### Set optimizer ######')
    if args.backbone == 'ConvNet':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif 'ResNet' in args.backbone:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    elif args.backbone == 'AmdimNet':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Encoder')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  
    

    print('###### Training ######')
    global_count = 0
    real_way = args.way

    train_label = torch.arange(0, args.train_way, 1 / args.query).long().cuda() 
    label = torch.arange(0, args.way, 1 / args.query).long().cuda()

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        train_loss = Averager()
        train_acc = Averager()
        args.way = args.train_way
        
        train_batches = tqdm.tqdm(train_loader)
        for batch in train_batches:
            global_count += 1
            data = batch[0].cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, train_label)
            acc = count_acc(logits, train_label)

            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            train_batches.set_description(f'Training   | Epoch {epoch} | loss={loss.item():.4f} | acc={acc:.4f} |')

            train_loss.add(loss.item())
            train_acc.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        ###### Validation step ###########
        model.eval()
        val_loss = Averager()
        val_acc = Averager()
        args.way = real_way
        
        with torch.no_grad():
            val_batches = tqdm.tqdm(val_loader)
            for batch in val_batches:
                data = batch[0].cuda()
                logits = model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)    
                val_loss.add(loss.item())
                val_acc.add(acc)
                val_batches.set_description(f'Validation | Epoch {epoch} | loss={val_loss.item():.4f} | acc={val_acc.item():.4f} |')

        val_loss = val_loss.item()
        val_acc = val_acc.item()
        writer.add_scalar('data/val_loss', float(val_loss), epoch)
        writer.add_scalar('data/val_acc', float(val_acc), epoch)        

        if val_acc > explog.max_acc:
            explog.max_acc = val_acc
            explog.max_acc_epoch = epoch
            torch.save(dict(params=model.state_dict()), osp.join(args.save_path, 'max_acc.pth'))
            print(f'-------- New best epoch: {explog.max_acc_epoch} | Best val acc={explog.max_acc:.4f} --------')

        print(f'---------------------------------------------------------')

        explog.train_loss.append(train_loss.item())
        explog.train_acc.append(train_acc.item())
        explog.val_loss.append(val_loss)
        explog.val_acc.append(val_acc)

        explog.save(args.save_path)
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, 'epoch-last.pth'))

        lr_scheduler.step()
        
    writer.close()
    explog.save_json(args.save_path)


    print('###### Testing ######')
    test_set = Dataset('test', args)
    sampler = FewShotSampler(test_set.label, args.test_epi, args.way, args.shot, args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=2, pin_memory=True)
    test_acc_record = np.zeros((args.test_epi,))

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    model.eval()

    ave_acc = Averager()

    args.save_features = True
    init_saving_features(args)

    if 'ICN' in args.modules:
        args.save_icn_scores = True
        init_saving_icn_scores(args)

    with torch.no_grad():
        test_batches = tqdm.tqdm(loader)
        for i, batch in enumerate(test_batches, 1):
            data = batch[0].cuda()
            logits = model(data)
            acc = count_acc(logits, label)

            if args.save_features:
                args.fts_labels.append(batch[1][:args.way*args.shot])
                args.fts_ids.append(batch[2][:args.way*args.shot])
            
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            explog.test_acc.append(acc)

            test_batches.set_description(f'Testing | Acc={acc * 100} | Avg acc={ave_acc.item() * 100} |')
        
    m, pm = compute_confidence_interval(test_acc_record)
    print(f'Test Acc {m:.4f} + {pm:.4f}')
    explog.mean_acc = m

    elapsed_time = timer.stop()
    explog.elapsed_time = elapsed_time


    print('###### Saving logs ######')
    explog.save(args.save_path)
    explog.save_json(args.save_path)
    explog.save_csv(args.save_path)

    if args.save_features:
        explog.save_features(args.save_path)

    if 'ICN' in args.modules and args.save_icn_scores:
        explog.save_icnn_scores(args.save_path)
    
    print(f"Elapsed time: {elapsed_time}")