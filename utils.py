# -*- coding: utf-8 -*-
import json
import os
import shutil
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

class Clock(object):
    def __init__(self):
        self.begin = time.time()

    def step(self):
        self.now = time.time()
        return self.now - self.begin

def save_model(model, optim, scheduler, epoch, path):
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint ={
        'model': model_state_dict,
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)

def resume_model(model, optim, scheduler, path):
    checkpoint = torch.load(path)
    model_state_dict = checkpoint['model']

    # load model param
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    del model_state_dict

    # load optim and scheduler
    optim.load_state_dict(checkpoint['optim'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    resume_epoch = checkpoint['epoch']

    return resume_epoch

def load_model(model, path):
    checkpoint = torch.load(path)
    model_state_dict = checkpoint['model']
    epoch = checkpoint['epoch']

    # load model param
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    del model_state_dict
    return epoch

def plot_fig(wav_input, fn, sr=8000):
    plt.figure()
    plt.specgram(wav_input, Fs=sr)
    plt.ylabel('F[Hz]')
    plt.xlabel('T[s]')

    plt.savefig(fn)
    plt.close()

def plot_and_compare(wav_input, wav_target, wav_pred, fn, sr=8000):
    plt.figure()
    plt.subplot(311)
    plt.specgram(wav_input, Fs=sr)
    plt.ylabel('F[Hz]')
    plt.xlabel('T[s]')

    plt.subplot(312)
    plt.specgram(wav_target, Fs=sr)
    plt.ylabel('F[Hz]')
    plt.xlabel('T[s]')

    plt.subplot(313)
    plt.specgram(wav_pred, Fs=sr)
    plt.ylabel('F[Hz]')
    plt.xlabel('T[s]')

    plt.savefig(fn)
    plt.close()


def SNR_db_to_scale(db):
    return 1 / 10 ** (db / 20.0)

def scale_to_SNR_db(interf_scale):
    return 20 * np.log10(1 / interf_scale)

def getFnFromPath(path):
    return os.path.split(path)[-1]

def getListfromDir(path):
    outputList = []
    fn_list = os.listdir(path)
    for fn in fn_list:
        outputList.append(os.path.join(path, fn))
    return outputList

# https://github.com/pytorch/pytorch/issues/2830
def to_cuda(m, cuda):
    if cuda and torch.cuda.is_available():
        for state in m.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def concatenateFeature(inputList, dim):
    out = inputList[0]
    for i in range(1, len(inputList)):
        out = torch.cat((out, inputList[i]), dim=dim)
    return out


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, 'r')))

if __name__ == '__main__':
    clock = Clock()
    for i in range(0, 10):
        time.sleep(1)
        print(clock.step())

