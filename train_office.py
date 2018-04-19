import argparse
import pprint

import chainer
from chainer import training
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainer.training.triggers import MaxValueTrigger
import numpy as np
import matplotlib
matplotlib.use('agg')  # noqa: E402

import datasets
import office
import utils
from utils import setup_optimizer
from utils import LossAndAccuracy
from office_updater import Updater


def prepare_data(args):
    print("Begin loading source domain")
    source = datasets.Domain(args.source)
    print("Finish loading source domain")
    print("Begin loading target domain")
    target = datasets.Domain(args.target)
    print("Finish loading target domain")
    s_train = TransformDataset(source.train, utils.train_transform_office)
    t_train = TransformDataset(target.train, utils.train_transform_office)
    s_test = TransformDataset(source.test, utils.test_transform_office)
    t_test = TransformDataset(target.test, utils.test_transform_office)
    s_train = utils.data2iterator(s_train, args.batchsize, is_train=True,
                                  multiprocess=args.multiprocess)
    t_train = utils.data2iterator(t_train, args.batchsize, is_train=True,
                                  multiprocess=args.multiprocess)
    s_test = utils.data2iterator(s_test, args.batchsize, is_train=False,
                                  multiprocess=args.multiprocess)
    t_test = utils.data2iterator(t_test, args.batchsize, is_train=False,
                                  multiprocess=args.multiprocess)
    return s_train, t_train, s_test, t_test


def main(args):
    print("Begin data preparation.")
    s_train, t_train, s_test, t_test = prepare_data(args)
    print("Finish data preparation.")
    print("Begin building models.")
    encoder = office.Encoder()
    chainer.serializers.load_npz('alexnet.npz', encoder, strict=False)
    bottleneck = office.Bottleneck()
    classifier = office.Classifier()
    do_classifier = office.DomainClassifier()
    print("Finish building models.")

    encoder_opt = setup_optimizer(encoder, args.optimizer, args.lr)
    bottleneck_opt = setup_optimizer(bottleneck, args.optimizer, 10*args.lr)
    classifier_opt = setup_optimizer(classifier, args.optimizer, 10*args.lr)
    do_classifier_opt = setup_optimizer(do_classifier, args.optimizer, 10*args.lr)
    optimizers = {'encoder': encoder_opt,
                  'domain_classifier': do_classifier_opt,
                  'bottleneck': bottleneck_opt,
                  'classifier': classifier_opt}
    loss_list = ['loss_cla_s', 'loss_cla_t', 'loss_do_cla']
    target_model = LossAndAccuracy(encoder, classifier, bottleneck)

    updater = Updater(s_train, t_train, optimizers, args)
    out_dir = utils.prepare_dir(args)
    trainer = training.Trainer(updater, (args.max_iter, 'iteration'),
                               out=out_dir)
    trainer.extend(extensions.LogReport(trigger=(args.interval, args.unit)))
    for name, opt in optimizers.items():
        trainer.extend(
            extensions.snapshot_object(opt.target, filename=name),
            trigger=MaxValueTrigger('acc_t', (args.interval, args.unit)))
    trainer.extend(extensions.Evaluator(t_test, target_model,
                                        device=args.device), trigger=(args.interval, args.unit))
    trainer.extend(extensions.PrintReport([args.unit, *loss_list, 'acc_s', 'acc_t', 'elapsed_time']))
    trainer.extend(extensions.PlotReport([*loss_list], x_key=args.unit, file_name='loss.png', trigger=(args.interval, args.unit)))
    trainer.extend(extensions.PlotReport(['acc_s', 'acc_t'], x_key=args.unit, file_name='accuracy.png', trigger=(args.interval, args.unit)))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    print("Start training loops.")
    trainer.run()
    print("Finish training loops.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-g", type=int, default=0)
    parser.add_argument("--max_iter", "-i", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--batchsize", "-b", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default='result')
    parser.add_argument("--config_file", type=argparse.FileType(mode='r'))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", "-w", type=float, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--source", type=str, default="amazon")
    parser.add_argument("--target", type=str, default="webcam")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--multiprocess", type=int, default=0)
    parser.add_argument("--grl_max_iter", type=int, default=10000)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--mirror", type=int, default=0)
    parser.add_argument("--do_weight", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--grayscale", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="MomentumSGD")
    parser.add_argument("--training_mode", type=str, default="dann")
    parser.add_argument("--unit", type=str, choices=['iteration', 'epoch'], default="iteration")
    args = parser.parse_args()
    args = utils.parse_args(args)
    pprint.pprint(vars(args))

    main(args)
