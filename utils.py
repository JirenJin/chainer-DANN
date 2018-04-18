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
        with self.init_scope:
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


def transform_factory(size=None, pixel_mean=None, grayscale=False, scale=1.0,
                      crop_size=None, mirror=False, is_train=False):
    def transform(in_data):
        img, label = in_data
        if grayscale:
            # in case the img is already grayscale
            if img.shape[0] != 1:
                img = (img[0] * 0.2989 + img[1] * 0.5870 +
                       img[2] * 0.1140).reshape(1, img.shape[1], img.shape[2])
        if pixel_mean is not None:
            img -= pixel_mean
        if size is not None:
            # avoid unnecessary computation
            if (size, size) != img.shape[1:]:
                img = resize(img, (size, size))
        if crop_size is not None:
            if is_train:
                img = random_crop(img, (crop_size, crop_size))
            else:
                img = center_crop(img, (crop_size, crop_size))
        if mirror and is_train:
            img = random_flip(img, x_random=True)
        img *= scale
        return img, label
    return transform
