import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..'))
DATASETS_PATH = osp.normpath(osp.join(ROOT_PATH, 'datasets'))
SPLIT_PATH = osp.normpath(osp.join(ROOT_PATH, 'datasets/CrossKidneys'))

class CrossKidneys(Dataset):
    """ Usage: 
    """
    def __init__(self, setname, args):
        if setname == 'train':
            ds = args.trainset
        elif setname == 'val':
            ds = args.valset
        elif setname == 'test':
            ds = args.testset

        ds = ds.split(',')
        datasets = []
        for dataset in ds:
            if dataset == 'Cross':
                cross_ds = [f'MiniImagenet', f'CUB', f'CropDisease', f'Eurosat', f'ISIC']
                datasets += cross_ds
            else:
                datasets.append(dataset)

        data = []
        label = []
        lb = -1
        wnids = []

        for dataset in datasets:

            split_name = f'{dataset}_{setname}'
            if dataset == 'Daudon' or dataset == 'Estrade' or dataset == 'Elbeze':
                split_name = f'{dataset}_{args.ks_set}'

            csv_path = osp.join(SPLIT_PATH, f'{split_name}.csv')
            lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

            for l in lines:
                name, wnid = l.split(',')
                path = osp.join(DATASETS_PATH, dataset, 'images', name)
                if wnid not in wnids:
                    wnids.append(wnid)
                    lb += 1
                data.append(path)
                label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if args.backbone == 'ConvNet':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif 'ResNet' in args.backbone:
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif args.backbone == 'AmdimNet':
            INTERP = 3
            post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        img_id = path.split('/')[-1].split('.')[0]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, img_id

