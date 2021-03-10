import torch
import torch.nn as nn
import numpy as np
from util.metric import euclidean_dist
import torch.nn.functional as F
    
class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'ConvNet':
            hdim = 64
            from networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone == 'ResNet':
            hdim = 640
            from networks.resnet import ResNet as ResNet
            self.encoder = ResNet()
        elif args.backbone == 'AmdimNet':
            from networks.amdimnet import AmdimNet as AmdimNet
            self.encoder = AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
            hdim = args.rkhs
        else:
            raise ValueError('')

        self.fc = nn.Linear(hdim, args.num_class)

    def forward(self, data, is_emb = False):
        out = self.encoder(data)
        if not is_emb:
            out = self.fc(out)
        return out
    
    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        
        query = self.encoder(data_query)
        logits = euclidean_dist(query, proto)
        return logits