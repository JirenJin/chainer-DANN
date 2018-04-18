import chainer
import chainer.links as L
import chainer.functions as F
from chainer import initializers
from chainer.backends import cuda


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
    def __init__(self, pixel_mean=0):
        self.pixel_mean = pixel_mean
        super().__init__()
        init_w = initializers.Normal(0.1)
        init_b = initializers.Constant(0.1)
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, 5, pad=2, initialW=init_w, initial_bias=init_b)
            self.conv2 = L.Convolution2D(32, 48, 5, pad=2, initialW=init_w, initial_bias=init_b)

    def __call__(self, x):
        with cuda.get_device_from_array(x):
            pixel_mean = cuda.to_gpu(self.pixel_mean)
        x -= pixel_mean
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.reshape(h, (-1, 7*7*48))
        return h


class Classifier(chainer.Chain):
    def __init__(self):
        super().__init__()
        init_w = initializers.Normal(0.1)
        init_b = initializers.Constant(0.1)
        with self.init_scope():
            self.fc1 = L.Linear(7*7*48, 100, initialW=init_w, initial_bias=init_b)
            # self.fc2 = L.Linear(100, 100, initialW=init_w, initial_bias=init_b)
            self.fc3 = L.Linear(100, 10, initialW=init_w, initial_bias=init_b)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        # h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h


class DomainClassifier(chainer.Chain):
    def __init__(self):
        super().__init__()
        init_w = initializers.Normal(0.1)
        init_b = initializers.Constant(0.1)
        with self.init_scope():
            self.fc1 = L.Linear(7*7*48, 100, initialW=init_w, initial_bias=init_b)
            self.fc2 = L.Linear(100, 1, initialW=init_w, initial_bias=init_b)

    def __call__(self, x, scale=1.0):
        h = GRL(scale)(x)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h
