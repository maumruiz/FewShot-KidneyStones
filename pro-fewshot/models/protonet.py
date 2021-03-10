import torch
import torch.nn as nn
from util.metric import euclidean_dist

class ProtoNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'ConvNet':
            from networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone == 'ResNet':
            from networks.resnet import ResNet
            self.encoder = ResNet()
        elif args.backbone == 'AmdimNet':
            from networks.amdimnet import AmdimNet
            self.encoder = AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
        else:
            raise ValueError('Encoder doesnt exist')

    def forward(self, data):
        embeddings = self.encoder(data)

        supp_fts = embeddings[:self.args.way*self.args.shot]
        query_fts = embeddings[self.args.way*self.args.shot:]

        # Flatten the features
        supp_fts = supp_fts.reshape(self.args.shot*self.args.way, -1)
        query_fts = query_fts.reshape(self.args.query*self.args.way, -1)

        # proto = supp_fts.reshape(self.args.way, self.args.shot, -1).mean(dim=1)
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