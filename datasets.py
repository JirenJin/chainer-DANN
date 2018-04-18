import copy
import sys
import pickle as pkl

import scipy
import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainercv.datasets import DirectoryParsingLabelDataset


class Domain(object):
    """Dataset class specifically for a single domain.
    Including training set and testing set for the domain.
    """
    _num_dict = {
        'mnist': 10,
        'mnistm': 10,
        'mnist2000': 10,
        'usps1800': 10,
        'svhn': 10,
        'amazon': 31,
        'dslr': 31,
        'webcam': 31,
        'visda_train': 10,
        'visda_validation': 10,
        }

    def __init__(self, name, seed=None):
        self.name = name
        self.seed = seed if seed else np.random.randint(10000)
        np.random.seed(self.seed)
        self.train, self.test = self._create_dataset()

    def __repr__(self):
        return (f'Domain(name={self.name!r}, '
                f'num_classes={self.num_classes!r}, '
                f'seed={self.seed!r})')

    def _create_dataset(self):
        name = self.name
        seed = self.seed
        if name == 'mnist':
            train, test = chainer.datasets.get_mnist(ndim=3, rgb_format=True)
        elif name == 'usps1800':
            train, test = get_usps1800(seed=seed)
        elif name == 'mnist2000':
            train, test = get_mnist2000(seed=seed)
        elif name == 'svhn':
            train, test = chainer.datasets.get_svhn()
        elif name == 'mnistm':
            train, test = get_mnistm()
        elif name == 'amazon':
            train = DirectoryParsingLabelDataset('./data/Office/amazon/images')
            test = copy.deepcopy(train)
        elif name == 'dslr':
            train = DirectoryParsingLabelDataset('./data/Office/dslr/images')
            test = copy.deepcopy(train)
        elif name == 'webcam':
            train = DirectoryParsingLabelDataset('./data/Office/webcam/images')
            test = copy.deepcopy(train)
        elif name == 'visda_train':
            train = DirectoryParsingLabelDataset('./data/VisDA/train')
            test = copy.deepcopy(train)
        elif name == 'visda_validation':
            train = DirectoryParsingLabelDataset('./data/VisDA/validation')
            test = copy.deepcopy(train)
        else:
            sys.exit("The domain name {} is wrong.".format(self.name))
        return train, test

    def sample_visualize(self, num, dpi=100):
        perm = np.random.permutation(len(self.train))[:num]
        imgs = np.stack(self.train[index][0] for index in perm)
        if imgs.max() > 1:
            imgs /= 255
        n, c, h, w = imgs.shape
        x_plots = np.ceil(np.sqrt(n))
        y_plots = x_plots
        plt.figure(figsize=(w*x_plots/dpi, h*y_plots/dpi), dpi=dpi)
        for i, img in enumerate(imgs):
            plt.subplot(y_plots, x_plots, i+1)

            if c == 1:
                plt.imshow(img[0], cmap=plt.cm.gray)
            else:
                plt.imshow(img.transpose((1, 2, 0)), interpolation="nearest")

            plt.axis('off')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,
                                hspace=0)

    @property
    def num_classes(self):
        return self._num_dict[self.name]


class MNISTMTrain(chainer.dataset.DatasetMixin):
    def __init__(self):
        imgs = pkl.load(open('data/MNISTM/mnistm_data.pkl', 'rb'))
        train_imgs = np.transpose(
            imgs['train'], (0, 3, 1, 2)).astype('f') / 255
        train_labels = pkl.load(open(
            'data/MNISTM/mnistm_train_labels.pkl', 'rb')).astype('i')
        self.values = list(zip(train_imgs, train_labels))

    def __len__(self):
        return len(self.values)

    def get_example(self, i):
        return self.values[i]


class MNISTMTest(chainer.dataset.DatasetMixin):
    def __init__(self):
        imgs = pkl.load(open('data/MNISTM/mnistm_data.pkl', 'rb'))
        test_imgs = np.transpose(imgs['test'], (0, 3, 1, 2)).astype('f') / 255
        test_labels = pkl.load(open(
            'data/MNISTM/mnistm_test_labels.pkl', 'rb')).astype('i')
        self.values = list(zip(test_imgs, test_labels))

    def __len__(self):
        return len(self.values)

    def get_example(self, i):
        return self.values[i]


def get_mnistm():
    return MNISTMTrain(), MNISTMTest()


class USPS1800(chainer.dataset.DatasetMixin):
    def __init__(self, seed=1234):
        usps = scipy.io.loadmat('data/usps_all.mat')
        usps_imgs = np.vstack(usps['data'].transpose(2, 1, 0))
        usps_imgs = np.reshape(usps_imgs, (11000, 1, 16, 16)) / 255
        usps_imgs = np.transpose(usps_imgs, (0, 1, 3, 2))
        usps_labels = np.concatenate(
            np.outer(np.roll(np.arange(10), -1), np.ones(1100))).astype('i')
        np.random.seed(seed)
        perm = np.random.permutation(11000)[:1800]
        sub_imgs = usps_imgs[perm].astype('f')
        sub_labels = usps_labels[perm]
        self.values = list(zip(sub_imgs, sub_labels))

    def __len__(self):
        return len(self.values)

    def get_example(self, i):
        return self.values[i]


def get_usps1800(seed=1234):
    return USPS1800(seed), USPS1800(seed)


def get_mnist2000(seed=1234):
    train, test = chainer.datasets.get_mnist(ndim=3, rgb_format=False)
    all_data = chainer.datasets.ConcatenatedDataset(train, test)
    mnist2000, _ = chainer.datasets.split_dataset_random(all_data,
                                                         2000, seed=seed)
    return mnist2000, mnist2000
