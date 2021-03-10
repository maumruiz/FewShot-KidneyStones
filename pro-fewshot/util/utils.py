import os
import shutil
import time
import random
import torch

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x

def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove:
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.makedirs(path)

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
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

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
