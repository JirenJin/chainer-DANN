import os
import datetime
import json
from functools import wraps
from copy import deepcopy

import chainer
import chainer.functions as F
from chainercv.transforms import center_crop
from chainercv.transforms import random_crop
from chainercv.transforms import random_flip
from chainercv.transforms import resize
import yaml
import numpy as np


def setup_optimizer(model, opt_name, lr):
    optimizer = chainer.optimizers.__dict__[opt_name](lr)
    optimizer.setup(model)
    return optimizer


def parse_args(args):
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        for key, value in data.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print("{} is ignored, please check the key name.".format(key))
    return args


class LossAndAccuracy(chainer.Chain):
    def __init__(self, encoder, classifier):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.classifier = classifier

    def __call__(self, x, t):
        logits = self.classifier(self.encoder(x))
        loss = F.softmax_cross_entropy(logits, t)
        accuracy = F.accuracy(logits, t)
        chainer.report({'loss_cla_t': loss})
        chainer.report({'acc_t': accuracy})
        return loss


class Averager(object):
    def __init__(self):
        self.value = 0
        self.count = 0

    def add(self, x, num):
        self.value += x * num
        self.count += num

    @property
    def ave(self):
        return float(self.value) / self.count


def data2iterator(data, batchsize, is_train=True, multiprocess=0):
    if multiprocess:
        iterator = chainer.iterators.MultiprocessIterator(
            data, batchsize, shuffle=is_train, repeat=is_train,
            n_processes=multiprocess)
    else:
        iterator = chainer.iterators.SerialIterator(
            data, batchsize, shuffle=is_train, repeat=is_train)
    return iterator


def prepare_dir(args):
    # customize the output path
    date = datetime.datetime.now()
    date_str = date.strftime("%m%d%H%M%S")
    out_path = os.path.join(args.out, args.source + '_' + args.target, args.training_mode, date_str)
    try:
        os.makedirs(out_path)
    except OSError:
        pass
    # save all options for the experiment
    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f)
    return out_path



def train_transform_office(in_data):
    mean = np.load('imagenet_mean.npy').reshape(3, 256, 256)
    crop_size = 227

    img, label = in_data
    # subtract the mean file
    img = img - mean
    # random crop image to 227x227
    img = random_crop(img, (crop_size, crop_size))
    # random mirror the image
    img = random_flip(img, x_random=True)
    return img, label


def test_transform_office(in_data):
    mean = np.load('imagenet_mean.npy').reshape(3, 256, 256)
    crop_size = 227

    img, label = in_data
    # subtract the mean file
    img = img - mean
    # center crop image to 227x227
    img = center_crop(img, (crop_size, crop_size))
    return img, label
