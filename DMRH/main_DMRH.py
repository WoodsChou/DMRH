# main_DMRH.py
# coding=utf-8

import torch
import os

from DMRH.models import Model
from DMRH.configs import Config
from DMRH.utils import compress_hash, hamming_dist, get_data, calculate_map

# hyper-parameters
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # set the gpu


def train(opt):
    net = Model(opt.NBIT, opt.PRE_TRAIN_MODEL)
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.LR, momentum=opt.MOMENTUM, weight_decay=opt.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.LR_DECAY_EPOCH, opt.LR_DECAY_RATE)

    H = torch.zeros(opt.NUM_TR, opt.NBIT)

    net.train()
    usable_gpu = opt.USE_GPU and torch.cuda.is_available()
    if usable_gpu:
        net = net.cuda()
        print('GPU-ID:', os.environ["CUDA_VISIBLE_DEVICES"])

    _zero = torch.zeros(1)
    if usable_gpu:
        H = H.cuda()
        _zero = _zero.cuda()
        opt.TRAIN_LABELS = opt.TRAIN_LABELS.cuda()

    for epoch in range(opt.EPOCH):
        total_loss = 0
        for steps, (inputs, labels, index) in enumerate(opt.TRAIN_LOADER):
            _labels = opt.TRAIN_LABELS[index]
            if usable_gpu:
                inputs = inputs.cuda()

            _U = net(inputs)
            _H = _U.mean(2)
            _B = 2 * torch.gt(_H, 0).float() - 1
            H[index] = _H.data  # shape(BATCH_SIZE, NBIT)

            S = _labels.mm(opt.TRAIN_LABELS.t())  # shape(BATCH_SIZE, NUM_TR)
            S = torch.gt(S, 0).float()

            PHI = 0.5 * _H.mm(H.t())  # shape(BATCH_SIZE, NUM_TR)
            similarity = (-S * PHI + (1 + (-PHI.abs()).exp()).log() + PHI.max(_zero)).mean()

            regularization = (_B - _H).pow(2).mean()

            J = similarity + opt.ETA * regularization

            optimizer.zero_grad()
            J.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), opt.CLIP)
            optimizer.step()

            total_loss += J.item()
        scheduler.step()  # update the learning rate
        torch.cuda.empty_cache()
        print('Epoch:', epoch + 1, '| loss =', total_loss)

    torch.save(net.state_dict(), opt.NET_SAVE_NAME)
    return net, usable_gpu


def evaluate(opt):
    net = Model(opt.NBIT)
    net.load_state_dict(torch.load(opt.NET_SAVE_NAME))
    usable_gpu = opt.USE_GPU and torch.cuda.is_available()
    if usable_gpu:
        net = net.cuda()
    return net, usable_gpu


def computeMAP(opt, net, usable_gpu, top=5000):
    net.eval()

    I_query_B, L_te = get_data(opt, net, opt.TEST_LOADER2, compress_hash, usable_gpu)
    I_retri_B, L_re = get_data(opt, net, opt.RETRI_LOADER2, compress_hash, usable_gpu)

    dist = hamming_dist(I_query_B, I_retri_B)

    mAP = calculate_map(opt, dist, L_te, L_re, top)
    return mAP


if __name__ == '__main__':
    NBIT = 64
    npatch = 2

    # dataset, {0: cifar10, 1: nus-wide, 2: ms-coco}
    data_selected = 0

    if data_selected == 0: data_name = Config.CIFAR10
    elif data_selected == 1: data_name = Config.NUS
    else: data_name = Config.COCO

    net_save_name = './results/' + data_name + '/DMRH_patch' + str(npatch) + '_nbit' + str(NBIT) + '.pkl'
    opt = Config(data_name=data_name, data_path='./data/', nbit=NBIT, net_save_name=net_save_name,  npatch=npatch, epoch=9, lr_decay_epoch=3)

    print('dataset:', data_name, '| patch:', npatch, '| save_name:', net_save_name)

    TRAIN = True
    if TRAIN:
        net, use_gpu = train(opt)
        mAP = computeMAP(opt, net, use_gpu, 5000)
        print(mAP)
    else:
        net, use_gpu = evaluate(opt)
        mAP = computeMAP(opt, net, use_gpu, 5000)
        print(mAP)
