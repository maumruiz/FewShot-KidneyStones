import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class FewShotSampler():
    def __init__(self, label, n_episodes, n_way, n_shot, n_queries):
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_queries = n_queries

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        for i_batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]

            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_shot]
                for im in l[pos]:
                    batch.append(im)

            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_queries]
                for im in l[pos]:
                    batch.append(im)

            batch = torch.stack(batch)
            yield batch

