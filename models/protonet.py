import torch
import torch.nn as nn
from util.metric import euclidean_dist
import util.globals as globals

class ProtoNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'ConvNet':
            from networks.convnet import ConvNet
            self.encoder = ConvNet()
            hdim = 64
        elif args.backbone == 'ResNet12':
            from networks.resnet12 import ResNet
            self.encoder = ResNet()
            hdim = 512
        elif args.backbone == 'ResNet18':
            from networks.resnet18 import ResNet
            self.encoder = ResNet()
            hdim = 640
        elif args.backbone == 'AmdimNet':
            from networks.amdimnet import AmdimNet
            self.encoder = AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
            hdim = args.rkhs
        else:
            raise ValueError('Encoder doesnt exist')

        if args.parallel:
            self.encoder = torch.nn.DataParallel(self.encoder)

    def forward(self, data):
        embeddings = self.encoder(data)

        supp_fts = embeddings[:self.args.way*self.args.shot]
        query_fts = embeddings[self.args.way*self.args.shot:]

        n_dim = supp_fts.shape[-1]
        n_supp = supp_fts.shape[0]
        n_qry = query_fts.shape[0]

        # Flatten the features with avg pooling
        supp_fts = nn.AvgPool2d(n_dim)(supp_fts).reshape(n_supp, -1)
        query_fts = nn.AvgPool2d(n_dim)(query_fts).reshape(n_qry, -1)

        # Save the features to later visualization
        if self.args.save_features:
            self.args.features.append(supp_fts.cpu().detach())

        prototypes = self.compute_prototypes(supp_fts, self.args.way, self.args.shot)
        logits = -euclidean_dist(query_fts, prototypes) / self.args.temperature
        
        return logits

    def compute_prototypes(self, support: torch.Tensor, k: int, n: int) -> torch.Tensor:
        """Compute class prototypes from support samples.

        # Arguments
            support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
                dimension.
            k: int. "k-way" i.e. number of classes in the classification task
            n: int. "n-shot" of the classification task

        # Returns
            class_prototypes: Prototypes aka mean embeddings for each class
        """
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        class_prototypes = support.reshape(k, n, -1).mean(dim=1)
        return class_prototypes