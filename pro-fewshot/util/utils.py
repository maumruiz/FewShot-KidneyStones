import os
import shutil
import time
import random
import torch

def set_gpu(x):
    print(f'### Setting gpu: {x}')
    os.environ['CUDA_VISIBLE_DEVICES'] = x

def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove:
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.makedirs(path)

def delete_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Timer:
    def __init__(self):
        self._start_time = time.perf_counter()

    def start(self):
        """Start a new timer"""
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        s = time.perf_counter() - self._start_time
        self._start_time = None

        s = int(s)
        m = int(0)
        h = int(0)

        if s > 59:
            m = s // 60
            s = s % 60

        if m > 59:
            h = m // 60
            m = m % 60

        # print(f"Elapsed time: {h:d}:{m:02d}:{s:02d}")
        return f'{h:d}:{m:02d}:{s:02d}'

    def elapsed(self):
        elapsed = time.perf_counter() - self._start_time
        return self._time_to_str(elapsed)

    def estimate(self, epoch, max_epoch):
        estimated_time = (time.perf_counter() - self._start_time) / epoch * max_epoch
        return self._time_to_str(estimated_time)

    def _time_to_str(self, t):
        if t >= 3600:
            return '{:.1f}h'.format(t / 3600)
        if t >= 60:
            return '{:.1f}m'.format(t / 60)
        return '{:.1f}s'.format(t)