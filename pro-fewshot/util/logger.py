import os.path as osp
import json
import torch

class ExpLogger():
    def __init__(self, args):
        self.args = vars(args)
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.max_acc = 0.0
        self.max_acc_epoch = 0
        self.test_acc = []
        self.mean_acc = ''
        self.parameters = 0
        self.elapsed_time = ''
    
    def _to_obj(self):
        log = {}
        log['args'] = self.args
        log['train_loss'] = self.train_loss
        log['val_loss'] = self.val_loss
        log['train_acc'] = self.train_acc
        log['val_acc'] = self.val_acc
        log['max_acc'] = self.max_acc
        log['max_acc_epoch'] = self.max_acc_epoch
        log['test_acc'] = self.test_acc
        log['mean_acc'] = self.mean_acc
        log['parameters'] = self.parameters
        log['elapsed_time'] = self.elapsed_time
        return log
    
    def save(self, path):
        log = self._to_obj()
        torch.save(log, osp.join(path, 'experimentObj'))

    def save_json(self, path):
        log = self._to_obj()

        with open(osp.join(path, 'experiment.json'), "w") as write_file:
            json.dump(log, write_file, indent=4)

