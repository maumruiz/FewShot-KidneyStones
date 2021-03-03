import os.path as osp
import argparse
import time
from util.utils import ensure_path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet'])
    parser.add_argument('--model', type=str, default='ProtoNet', choices=['ProtoNet'])
    parser.add_argument('--model_type', type=str, default='ConvNet', choices=['ConvNet', 'ResNet'])
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--train_epi', type=int, default=2)
    parser.add_argument('--val_epi', type=int, default=2)
    parser.add_argument('--test_epi', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=1)

    # MiniImageNet, ConvNet, './saves/initialization/miniimagenet/con-pre.pth'
    # MiniImageNet, ResNet, './saves/initialization/miniimagenet/res-pre.pth'
    # CUB, ConvNet, './saves/initialization/cub/con-pre.pth'
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='runs')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', default=1234)

    return parser.parse_args()

def process_args(args):
    gmt = time.localtime() 
    timestmp = f'{gmt.tm_year}{gmt.tm_mon:02d}{gmt.tm_mday:02d}{gmt.tm_hour:02d}{gmt.tm_min:02d}{gmt.tm_sec:02d}'
    save_path1 = f'{args.dataset}-{args.model}-{args.model_type}'
    # save_path2 = f'{args.way}way{args.shot}shot_{args.query}q_{args.lr}lr_{args.step_size}step_{args.gamma}g_{args.temperature}t'
    save_path2 = f'{args.way}way{args.shot}shot_{args.query}q_{timestmp}'
    args.save_path = osp.join(args.save_path, osp.join(save_path1, save_path2))
    ensure_path(args.save_path)

def print_args(args):
    for key, value in args.__dict__.items():
        print(f'{key}: {value}')