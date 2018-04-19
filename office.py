from functools import partial

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import initializers


class GRL(chainer.Function):
    """Gradient Reversal Layer."""
    def __init__(self, scale):
        self.scale = scale

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, gradients):
        gw, = gradients
        return -1. * self.scale * gw,


class Encoder(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        return h


class Bottleneck(chainer.Chain):
    def __init__(self):
        super().__init__()
        init_w = initializers.Normal(0.005)
        init_b = initializers.Constant(0.1)
        with self.init_scope():
            self.fc = L.Linear(4096, 256, initialW=init_w, initial_bias=init_b)

    def __call__(self, x):
        h = self.fc(x)
        return h


class Classifier(chainer.Chain):
    def __init__(self):
        super().__init__()
        init_w = initializers.Normal(0.01)
        init_b = initializers.Constant(0)
        with self.init_scope():
            self.fc1 = L.Linear(None, 31, initialW=init_w, initial_bias=init_b)

    def __call__(self, x):
        h = self.fc1(x)
        return h


class DomainClassifier(chainer.Chain):
    def __init__(self):
        super().__init__()
        init_w1 = chainer.initializers.Normal(0.01)
        init_w2 = chainer.initializers.Normal(0.3)
        with self.init_scope():
            self.fc1 = L.Linear(None, 1024, initialW=init_w1)
            self.fc2 = L.Linear(1024, 1024, initialW=init_w1)
            self.fc3 = L.Linear(1024, 1, initialW=init_w2)

    def __call__(self, x, scale=1.0):
        h = GRL(scale)(x)
        h = F.relu(self.fc1(h))
        h = F.dropout(h)
        h = F.relu(self.fc2(h))
        h = F.dropout(h)
        h = self.fc3(h)
        return h
