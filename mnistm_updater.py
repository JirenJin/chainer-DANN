import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer import Variable

from base_da_updater import BaseDAUpdater


class Updater(BaseDAUpdater):
    """DANN updater."""
    def __init__(self, s_iter, t_iter, optimizers, args):
        super().__init__(s_iter, t_iter, optimizers, device=args.device)
        self.grl_max_iter = args.grl_max_iter
        self.max_iter = args.max_iter
        self.do_weight = args.do_weight
        self.lr = args.lr
        # source-only training or DANN training
        self.training_mode = args.training_mode
        self.enc = optimizers['encoder'].target
        self.cla = optimizers['classifier'].target
        self.do_cla = optimizers['domain_classifier'].target

    def update_core(self):
        # convenient to avoid device related errors
        cuda.Device(self.device).use()

        # adjust the learning rate and scale for GRL
        xp = self.enc.xp
        p1 = float(self.iteration) / self.grl_max_iter
        scale = min(2. / (1. + xp.exp(-10. * p1, dtype='f')) - 1, 0.5)
        # p2 = float(self.iteration) / self.max_iter
        # lr = self.lr / (1. + 10 * p2)**0.75
        # for opt in self.optimizers.values():
            # # I think this is equal to `opt.lr = lr`
            # # but this is the safer way to make sure lr is changed
            # opt.hyperparam.lr = lr

        # get a minibatch
        s_batch = next(self.s_iter)
        t_batch = next(self.t_iter)
        s_imgs, s_labels = self.converter(s_batch, self.device)
        t_imgs, _ = self.converter(t_batch, self.device)

        # source domain classification forward pass
        s_encoding = self.enc(s_imgs)
        s_logits = self.cla(s_encoding)
        loss_cla_s = F.softmax_cross_entropy(s_logits, s_labels)
        acc_s = F.accuracy(s_logits, s_labels)

        # domain classification forward pass
        if self.training_mode == 'dann':
            t_encoding = self.enc(t_imgs)

            s_do_logits = self.do_cla(s_encoding, scale)
            t_do_logits = self.do_cla(t_encoding, scale)

            s_do_labels = Variable(xp.zeros(s_do_logits.shape, dtype='i'))
            t_do_labels = Variable(xp.ones(t_do_logits.shape, dtype='i'))

            loss_do_cla = F.sigmoid_cross_entropy(s_do_logits, s_do_labels)
            loss_do_cla += F.sigmoid_cross_entropy(t_do_logits, t_do_labels)

        else:
            loss_do_cla = 0

        loss_total = loss_cla_s + self.do_weight * loss_do_cla

        # begin backward pass

        # clear gradients first
        for opt in self.optimizers.values():
            opt.target.cleargrads()

        # compute gradients
        loss_total.backward()

        # update the parameters
        for opt in self.optimizers.values():
            opt.update()

        # report the values for logging
        chainer.reporter.report({'acc_s': acc_s})
        chainer.reporter.report({'loss_cla_s': loss_cla_s})
        chainer.reporter.report({'loss_do_cla': loss_do_cla})
