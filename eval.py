#!/usr/bin/env python3

import argparse

import numpy as np
import scipy.io as sio
from sklearn import metrics


def eval(Y, gt):
    Y = np.squeeze(Y)
    gt = np.squeeze(gt)
    ari = metrics.adjusted_rand_score(gt, Y)
    ami = metrics.adjusted_mutual_info_score(gt, Y)
    nmi = metrics.normalized_mutual_info_score(gt, Y)
    print('## ARI: %0.3f AMI: %0.3f NMI: %0.3f' % (ari, ami, nmi))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Eval')
    argparser.add_argument('--data', type=str, metavar='<path>', required=True)
    args = argparser.parse_args()
    data = sio.loadmat(args.data)
    print('RCC')
    eval(data['rcc'], data['Y'])
    print('RCCDR')
    eval(data['rccdr'], data['Y'])
