import os.path as osp
import json
import torch

class TrainingLogger():
    def __init__(self, args):
        self.args = vars(args)
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.max_acc = 0.0
        self.max_acc_epoch = 0
    
    def save(self, path):
        trobj = {}
        trobj['args'] = self.args
        trobj['train_loss'] = self.train_loss
        trobj['val_loss'] = self.val_loss
        trobj['train_acc'] = self.train_acc
        trobj['val_acc'] = self.val_acc
        trobj['max_acc'] = self.max_acc
        trobj['max_acc_epoch'] = self.max_acc_epoch

        torch.save(trobj, osp.join(path, 'trlog'))

        with open(osp.join(path, 'trlog.json'), "w") as write_file:
            json.dump(trobj, write_file, indent=4)

    def save_json(self, path):
        trobj = {}
        trobj['args'] = self.args
        trobj['train_loss'] = self.train_loss
        trobj['val_loss'] = self.val_loss
        trobj['train_acc'] = self.train_acc
        trobj['val_acc'] = self.val_acc
        trobj['max_acc'] = self.max_acc
        trobj['max_acc_epoch'] = self.max_acc_epoch

        with open(osp.join(path, 'experiment.json'), "w") as write_file:
            json.dump(trobj, write_file, indent=4)

