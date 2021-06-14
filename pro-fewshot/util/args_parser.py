import os.path as osp
import argparse
import time
from util.utils import ensure_path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet', 'KidneyStones', 'Cross'])
    parser.add_argument('--model', type=str, default='ProtoNet', choices=['ProtoNet', 'Classifier'])
    parser.add_argument('--modules', type=str)
    parser.add_argument('--backbone', type=str, default='ConvNet', choices=['ConvNet', 'ResNet12', 'ResNet18', 'AmdimNet'])
    parser.add_argument('--optimizer', type=str, default='recommended', choices=['recommended', 'Adam', 'SGD'])
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_epi', type=int, default=100)
    parser.add_argument('--val_epi', type=int, default=500)
    parser.add_argument('--test_epi', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--cross_ds', type=str)
    parser.add_argument('--model_name', type=str)

    # MiniImageNet, ConvNet, './saves/initialization/miniimagenet/con-pre.pth'
    # MiniImageNet, ResNet, './saves/initialization/miniimagenet/res-pre.pth'
    # CUB, ConvNet, './saves/initialization/cub/con-pre.pth'
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--save_path', type=str, default='runs')
    parser.add_argument('--save_features', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--tag', type=int, default=99)

    # AMDIM Modelrd
    parser.add_argument('--ndf', type=int, default=192)
    parser.add_argument('--rkhs', type=int, default=1536)
    parser.add_argument('--nd', type=int, default=8)

    # CTM args
    parser.add_argument('--ctm_blocks', type=int, default=4)
    parser.add_argument('--ctm_out_channels', type=int, default=0)
    parser.add_argument('--ctm_block_type', type=str, default='ConvBlock', choices=['ConvBlock', 'ResBlock'])
    parser.add_argument('--ctm_m_type', type=str, default='fused', choices=['fused', 'avg'])
    parser.add_argument('--ctm_reduce_dims', action='store_true')
    parser.add_argument('--ctm_split_blocks', action='store_true')

    # ICN args
    parser.add_argument('--save_icn_scores', action='store_true')
    parser.add_argument('--icn_models', type=str, default='pca,isomap')
    parser.add_argument('--icn_reduction_set', type=str, default='all', choices=['support', 'all'])
    parser.add_argument('--icn_reduction_type', type=str, default='unsupervised', choices=['supervised', 'unsupervised'])
    parser.add_argument('--icn_multiple_components', action='store_true')
    parser.add_argument('--icn_n_dims', type=int, default=6)
    parser.add_argument('--icn_original_score', action='store_true')

    return parser.parse_args()

def process_args(args):
    args.parallel = len(args.gpu.split(',')) > 1

    if args.train_way == 0:
        args.train_way = args.way

    if args.save_features:
        init_saving_features(args)

    args.modules = args.modules.split(',') if args.modules else []

    if 'CTM' not in args.modules:
        del args.ctm_blocks
        del args.ctm_out_channels
        del args.ctm_block_type
        del args.ctm_m_type
        del args.ctm_reduce_dims
        del args.ctm_split_blocks

    if 'ICN' not in args.modules:
        del args.save_icn_scores
        del args.icn_models
        del args.icn_reduction_set
        del args.icn_reduction_type
        del args.icn_multiple_components
        del args.icn_n_dims
        del args.icn_original_score
    else:
        args.icn_models = args.icn_models.split(',') if args.icn_models else []

    if args.backbone != 'AmdimNet':
        del args.ndf
        del args.rkhs
        del args.nd

    gmt = time.localtime() 
    # timestmp = f'{gmt.tm_year}{gmt.tm_mon:02d}{gmt.tm_mday:02d}{gmt.tm_hour:02d}{gmt.tm_min:02d}{gmt.tm_sec:02d}'
    timestmp = f'{gmt.tm_mon:02d}{gmt.tm_mday:02d}-{gmt.tm_hour:02d}{gmt.tm_min:02d}{gmt.tm_sec:02d}'
    path1 = [args.dataset, args.model, args.backbone] + args.modules
    save_path1 = "-".join(path1)
    save_path2 = f'{args.way}way{args.shot}shot_{args.tag:02d}_{timestmp}'
    args.save_path = osp.join(args.save_path, osp.join(save_path1, save_path2))
    ensure_path(args.save_path)

def print_args(args):
    for key, value in args.__dict__.items():
        print(f'{key}: {value}')

def init_saving_features(args):
    args.features = []
    args.fts_ids = []
    args.fts_labels = []

def init_saving_icn_scores(args):
    args.icn_log = {}
    args.icn_log['original'] = []
    args.icn_log['best'] = []