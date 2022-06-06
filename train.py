import os.path as osp

import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader.samplers import FewShotSampler
from util.utils import set_gpu, Averager, Timer, set_seed, delete_path
from util.metric import compute_confidence_interval, count_acc
from util.args_parser import get_args, process_args, print_args, init_saving_features
from util.logger import ExpLogger
import util.globals as glob


def main(args):

    set_seed(args.seed)
    set_gpu(args.gpu)
    cudnn.benchmark = True

    timer = Timer()
    timer.start()

    explog = ExpLogger(args)
    writer = SummaryWriter(log_dir=args.save_path)

    glob.init()

    print('-- Loading data --')
    if args.dataset == 'MiniImageNet':
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'KidneyStones':
        from dataloader.kidney_stones import KidneyStones as Dataset
    elif args.dataset == 'Cross':
        from dataloader.cross import Cross as Dataset
    elif args.dataset == 'CrossKidneys':
        from dataloader.cross_kidneys import CrossKidneys as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args)
    train_sampler = FewShotSampler(trainset.label, args.train_epi, args.train_way, args.shot, args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = FewShotSampler(valset.label, args.val_epi, args.way, args.shot, args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)
    

    print('-- Creating model --')
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
            # pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # print('Pretrained dict keys:')
            # print(pretrained_dict.keys())
        elif 'model_sd' in model_detail:
            pretrained_dict = model_detail['model_sd']
            # remove weights for FC
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print('Pretrained dict keys:')
            print(pretrained_dict.keys())
        else:
            pretrained_dict = model_detail['model']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
        model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda()

    explog.parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('-- Setting optimizer --')
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Optimizer')

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  
    

    print(f'###### Training experiment {args.tag} ######')
    # torch.autograd.set_detect_anomaly(True)
    max_acc_model = None
    real_way = args.way

    train_label = torch.arange(0, args.train_way, 1 / args.query).long().cuda() 
    label = torch.arange(0, args.way, 1 / args.query).long().cuda()

    timer_trainval = Timer()
    timer_epoch = Timer()

    for epoch in range(1, args.max_epoch + 1):
        timer_epoch.start()
        model.train()
        train_loss = Averager()
        train_acc = Averager()
        args.way = args.train_way
        
        train_batches = tqdm.tqdm(train_loader, dynamic_ncols=True, leave=False)
        for batch in train_batches:
            data = batch[0].cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, train_label)
            acc = count_acc(logits, train_label)

            writer.add_scalar('data/loss', float(loss), epoch)
            writer.add_scalar('data/acc', float(acc), epoch)
            train_batches.set_description(f'TRAIN | Epoch {epoch} | Loss={loss.item():.4f} | Acc={acc*100:.2f} |')

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
            val_batches = tqdm.tqdm(val_loader, dynamic_ncols=True, leave=False)
            for batch in val_batches:
                data = batch[0].cuda()
                logits = model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)

                val_loss.add(loss.item())
                val_acc.add(acc)
                val_batches.set_description(f'VAL   | Epoch {epoch} | Loss={val_loss.item():.4f} | Acc={val_acc.item()*100:.2f} |')

        val_loss = val_loss.item()
        val_acc = val_acc.item()
        writer.add_scalar('data/val_loss', float(val_loss), epoch)
        writer.add_scalar('data/val_acc', float(val_acc), epoch)

        log_str = f'Epoch {epoch:2d} | TRAIN (Loss: {train_loss.item():.4f}, Acc: {train_acc.item()*100:.2f}) | VAL (Loss: {val_loss:.4f}, Acc: {val_acc*100:.4f})'

        t_epoch = timer_epoch.elapsed()
        t_trainval = timer_trainval.elapsed()
        t_estimate = timer_trainval.estimate(epoch, args.max_epoch)
        log_str += f' | TIME (Epoch: {t_epoch}, Estimate: {t_trainval}/{t_estimate})'

        if val_acc > explog.max_acc:
            explog.max_acc = val_acc
            explog.max_acc_epoch = epoch
            torch.save(dict(params=model.state_dict()), osp.join(args.save_path, 'max_acc.pth'))
            max_acc_model = dict(params=model.state_dict())
            log_str += ' ---- NEW BEST EPOCH ----'

        print(log_str)

        explog.train_loss.append(train_loss.item())
        explog.train_acc.append(train_acc.item())
        explog.val_loss.append(val_loss)
        explog.val_acc.append(val_acc)

        # explog.save(args.save_path)
        # torch.save(dict(params=model.state_dict()), osp.join(args.save_path, 'epoch-last.pth'))

        lr_scheduler.step()
        
    writer.close()
    explog.save_json(args.save_path)

    if args.model_name:
        torch.save(max_acc_model, osp.join('pretrained', f'{args.model_name}.pth'))


    print('###### Testing ######')
    test_set = Dataset('test', args)
    sampler = FewShotSampler(test_set.label, args.test_epi, args.way, args.shot, args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=2, pin_memory=True)
    test_acc_record = np.zeros((args.test_epi,))

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    model.eval()

    ave_acc = Averager()

    if args.save_features:
        init_saving_features(args)

    with torch.no_grad():
        test_batches = tqdm.tqdm(loader, dynamic_ncols=True, leave=False)
        for i, batch in enumerate(test_batches, 1):
            data = batch[0].cuda()
            logits = model(data)
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc

            if args.save_features:
                args.fts_labels.append(batch[1][:args.way*args.shot])
                args.fts_ids.append(batch[2][:args.way*args.shot])

            if args.save_logits:
                labels = batch[1][:args.way*args.shot].unique_consecutive()
                pred = torch.argmax(logits, dim=1)
                explog.query_predictions += [labels[x].item() for x in pred]
                explog.query_labels += batch[1][args.way*args.shot:].tolist()
            
            explog.test_acc.append(acc)

            test_batches.set_description(f'Testing | Avg acc={ave_acc.item() * 100:.2f} |')
        
    m, pm = compute_confidence_interval(test_acc_record)
    print(f'TEST Acc {m:.4f} +- {pm:.4f}')
    explog.mean_acc = m

    elapsed_time = timer.stop()
    explog.elapsed_time = elapsed_time


    print('-- Saving logs --')
    explog.save(args.save_path)
    explog.save_json(args.save_path)
    explog.save_trainval(args.save_path)
    explog.save_test(args.save_path)
    explog.save_results(args.save_path)

    if args.save_features:
        explog.save_features(args.save_path)

    if args.save_logits:
        explog.save_logits(args.save_path)
    
    print(f"### Elapsed time: {elapsed_time}")

if __name__ == '__main__':
    print('###### Start experiment with args: ######')
    args = get_args()
    process_args(args)
    print_args(args)
    try:
        main(args)
    except Exception as inst:
        print('----------------------')
        print("Oops!! Something went wrong!")
        print(f'{type(inst).__name__}: {inst}')    # the exception instance
        delete_path(args.save_path)
        raise