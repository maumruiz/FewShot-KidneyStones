import numpy as np
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataloader.samplers import FewShotSampler
from util.utils import set_gpu, Averager, set_seed, delete_path
from util.metric import compute_confidence_interval, count_acc
from util.args_parser import get_args, process_args, print_args, init_saving_features
from util.logger import ExpLogger


def main(args):
    set_seed(args.seed)
    set_gpu(args.gpu)
    cudnn.benchmark = True

    explog = ExpLogger(args)

    print('-- Loading data --')
    if args.dataset == 'MiniImageNet':
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'KidneyStones':
        from dataloader.kidney_stones import KidneyStones as Dataset
    elif args.dataset == 'CrossKidneys':
        from dataloader.cross_kidneys import CrossKidneys as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    test_set = Dataset('test', args)
    sampler = FewShotSampler(test_set.label, args.test_epi, args.way, args.shot, args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    print('-- Creating model --')
    if args.model == 'ProtoNet':
        from models.protonet import ProtoNet as Model
    else:
        raise ValueError('Non-supported Model.')
    model = Model(args)
    
    model_detail = torch.load(args.init_weights)
    if 'params' in model_detail:
        model_dict = model_detail['params']
    else:
        model_dict = model.state_dict()
        pretrained_dict = model_detail['model']
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
        model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model = model.cuda()
    model.eval()

    model_name = args.init_weights.split('/')[1].split('.')[0]
    print(f"###### Testing: Model {model_name} | Shot: {args.shot} | Experiment {args.tag} ######")
    test_acc_record = np.zeros((args.test_epi,))
    ave_acc = Averager()
    label = torch.arange(0, args.way, 1 / args.query).long().cuda()

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
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
    explog.mean_acc = m

    print('-- Saving logs --')
    explog.save_json(args.save_path)
    explog.save_test(args.save_path)
    explog.save_results(args.save_path)

    if args.save_features:
        explog.save_features(args.save_path)

    if args.save_logits:
        explog.save_logits(args.save_path)

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