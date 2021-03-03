import os.path as osp

import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader.samplers import CategoriesSampler
from util.utils import set_gpu, Averager, Timer, set_seed
from util.metric import compute_confidence_interval, count_acc
from util.args_parser import get_args, process_args, print_args
from util.logger import TrainingLogger


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
    train_sampler = CategoriesSampler(trainset.label, args.train_epi, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_epi, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    

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


    print('###### Set optimizer ######')
    if args.model_type == 'ConvNet':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model_type == 'ResNet':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    elif args.model_type == 'AmdimNet':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Encoder')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  
    

    print('###### Training ######')
    trlog = TrainingLogger(args)
    global_count = 0
    writer = SummaryWriter(log_dir=args.save_path)
    
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        tl = Averager()
        ta = Averager()

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)
        
        train_batches = tqdm.tqdm(train_loader)
        for i, batch in enumerate(train_batches):
            global_count = global_count + 1
            data, _ = [b.cuda() for b in batch]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]
            logits = model(data_shot, data_query)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            train_batches.set_description(f'epoch {epoch}, loss={loss.item():.4f} acc={acc:.4f}')

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)
            
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]
    
                logits = model(data_shot, data_query)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)    
                vl.add(loss.item())
                va.add(acc)

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), epoch)
        writer.add_scalar('data/val_acc', float(va), epoch)        
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog.max_acc:
            trlog.max_acc = va
            trlog.max_acc_epoch = epoch
            torch.save(dict(params=model.state_dict()), osp.join(args.save_path, 'max_acc.pth'))

        trlog.train_loss.append(tl)
        trlog.train_acc.append(ta)
        trlog.val_loss.append(vl)
        trlog.val_acc.append(va)

        print(f'Best epoch: {trlog.max_acc_epoch} | Best val acc={trlog.max_acc:.4f}')

        trlog.save(args.save_path)
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, 'epoch-last.pth'))

        lr_scheduler.step()
    writer.close()
    trlog.save_json(args.save_path)


    print('###### Testing ######')
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_epi, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((args.test_epi,))

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    model.eval()

    ave_acc = Averager()
    label = torch.arange(args.way).repeat(args.query)
    label = label.type(torch.cuda.LongTensor)
        
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            data, _ = [b.cuda() for b in batch]
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]
    
            logits = model(data_shot, data_query)
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
    m, pm = compute_confidence_interval(test_acc_record)
    print(f'Val Best Acc: {trlog.max_acc:.4f} | Test Acc {ave_acc.item():.4f}')
    print(f'Test Acc {m:.4f} + {pm:.4f}')
    timer.stop()