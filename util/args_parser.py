import os.path as osp
import argparse
import time
from util.utils import ensure_path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet', 'KidneyStones', 'Cross', 'CrossKidneys'])
    parser.add_argument('--model', type=str, default='ProtoNet', choices=['ProtoNet', 'Classifier'])
    parser.add_argument('--backbone', type=str, default='AmdimNet', choices=['ConvNet', 'ResNet12', 'ResNet18', 'AmdimNet'])
    parser.add_argument('--optimizer', type=str, default='recommended', choices=['recommended', 'Adam', 'SGD'])
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_epi', type=int, default=100)
    parser.add_argument('--val_epi', type=int, default=500)
    parser.add_argument('--test_epi', type=int, default=1000)

    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=128)

    parser.add_argument('--cross_ds', type=str)
    parser.add_argument('--model_name', type=str)

    parser.add_argument('--trainset', type=str)
    parser.add_argument('--valset', type=str)
    parser.add_argument('--testset', type=str)
    parser.add_argument('--ks_set', type=str)

    # MiniImageNet, ConvNet, './saves/initialization/miniimagenet/con-pre.pth'
    # MiniImageNet, ResNet, './saves/initialization/miniimagenet/res-pre.pth'
    # CUB, ConvNet, './saves/initialization/cub/con-pre.pth'
    parser.add_argument('--init_weights', type=str, default=None)

    parser.add_argument('--save_path', type=str, default='runs')
    parser.add_argument('--save_features', action='store_true')
    parser.add_argument('--save_logits', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--tag', type=int, default=99)

    # AMDIM Modelrd
    parser.add_argument('--ndf', type=int, default=192)
    parser.add_argument('--rkhs', type=int, default=1536)
    parser.add_argument('--nd', type=int, default=8)

    return parser.parse_args()

def process_args(args):
    args.parallel = len(args.gpu.split(',')) > 1

    if args.train_way == 0:
        args.train_way = args.way

    if args.save_features:
        init_saving_features(args)

    if args.backbone != 'AmdimNet':
        del args.ndf
        del args.rkhs
        del args.nd

    if args.optimizer == 'recommended':
        if args.backbone == 'ConvNet':
            optimizer = 'Adam'
        elif 'ResNet' in args.backbone:
            optimizer = 'SGD'
        elif args.backbone == 'AmdimNet':
            optimizer = 'SGD'
        else:
            raise ValueError('No Such Encoder')
        args.optimizer = optimizer

    gmt = time.localtime() 
    # timestmp = f'{gmt.tm_year}{gmt.tm_mon:02d}{gmt.tm_mday:02d}{gmt.tm_hour:02d}{gmt.tm_min:02d}{gmt.tm_sec:02d}'
    timestmp = f'{gmt.tm_mon:02d}{gmt.tm_mday:02d}-{gmt.tm_hour:02d}{gmt.tm_min:02d}{gmt.tm_sec:02d}'
    path1 = [args.dataset, args.model, args.backbone]
    save_path1 = "-".join(path1)
    save_path2 = f'{args.way}way{args.shot}shot'
    save_path3 = f'{args.tag:02d}_{timestmp}'
    args.save_path = osp.join(args.save_path, osp.join(save_path1, save_path2, save_path3))
    ensure_path(args.save_path)

def print_args(args):
    args_str = ''
    i = 0
    for key, value in args.__dict__.items():
        if key == 'save_path':
            continue
        if i % 6 == 0 and i > 0:
            args_str += '\n'
        args_str += f'{key + ": " + str(value):21.21} || '
        i += 1
    print(args_str[:-3])
    if 'save_path' in args.__dict__.keys():
        print(f'save_path: {args.save_path}')

def init_saving_features(args):
    args.features = []
    args.fts_ids = []
    args.fts_labels = []