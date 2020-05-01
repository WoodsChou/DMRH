# train.py
# coding = utf-8

import argparse
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from DMRH.configs import Config
from DMRH.main_DMRH import train, evaluate, computeMAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DMRH')
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--npatch', type=int, default=2, help='hyper-parameter')
    parser.add_argument('--dataset', type=int, default=0, help='dataset')
    parser.add_argument('--nbit', type=int, default=64, help='length of hash codes')
    parser.add_argument('--is_train', type=bool, default=True, help='whether to train model')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    npatch = args.npatch
    NBIT = args.nbit

    # dataset, {0: cifar10, 1: nus-wide, 2: ms-coco}
    data_selected = args.dataset

    if data_selected == 0: data_name = Config.CIFAR10
    elif data_selected == 1: data_name = Config.NUS
    else: data_name = Config.COCO

    net_save_name = './results/' + data_name + '/DMRH_patch' + str(npatch) + '_nbit' + str(NBIT) + '.pkl'
    opt = Config(data_name=data_name, data_path='./data/', nbit=NBIT, net_save_name=net_save_name,  npatch=npatch, epoch=9, lr_decay_epoch=3)

    print('dataset:', data_name, '| patch:', npatch, '| save_name:', net_save_name)

    TRAIN = args.is_train
    if TRAIN:
        net, use_gpu = train(opt)
        mAP = computeMAP(opt, net, use_gpu, 5000)
        print(mAP)
    else:
        net, use_gpu = evaluate(opt)
        mAP = computeMAP(opt, net, use_gpu, 5000)
        print(mAP)

