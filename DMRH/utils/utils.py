# utils.py
# coding = utf-8

import numpy as np
import torch
import copy


# get the Hash
def compress_hash(net, I_data, usable_gpu):
    if usable_gpu:
        I_data = I_data.cuda()
    U = net(I_data)
    H = U.mean(2)
    B = (H >= 0).data.cpu().numpy().astype(int)
    return B


def hamming_dist(B1, B2):
    num_B1 = len(B1)
    num_B2 = len(B2)
    dist = np.zeros((num_B1, num_B2))
    for i in range(num_B1):
        dist[i] = np.sum(abs(B1[i] - B2), 1).reshape(-1)
    return dist


def get_data(opt, net, loader, hash_func, usable_gpu):
    B = []
    L = []
    for steps, (inputs, labels, index) in enumerate(loader):
        batch_size = labels.size(0)
        if opt.LABEL_IS_SINGLE:
            _labels = np.zeros((batch_size, opt.NUM_CA))
            for i in range(batch_size):
                _labels[i, labels[i]] = 1
        else:
            _labels = copy.deepcopy(labels)

        B.append(hash_func(net, inputs, usable_gpu))
        L.append(_labels)
    torch.cuda.empty_cache()
    B = np.concatenate(B)
    L = np.concatenate(L)
    return B, L


def calculate_map(opt, dist, L_te, L_re, top):
    mAP = 0
    for i in range(opt.NUM_TE):
        # find the category of the ith query
        kind = np.argwhere(L_te[i] == 1).reshape(-1)
        # find the data which have the same category with the query
        index = np.argwhere(np.sum(L_re[:, kind], 1) > 0).reshape(-1)

        pos = np.argsort(dist[i])
        element = np.in1d(pos[:top], index)
        loc = np.argwhere(element == True).reshape(-1)

        # compute the AP of the query and add to MAP
        if len(loc) > 0:
            mAP += np.sum(np.array(range(1, len(loc) + 1)) / (loc + 1)) / len(loc)
    mAP = mAP / opt.NUM_TE
    return mAP
