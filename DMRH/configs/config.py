# config.py
# coding = utf-8

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import warnings
import scipy.io as scio

from DMRH.dataset import _CIFAR10, _NUS, _COCO


class Config(object):
    CIFAR10 = 'cifar10'
    NUS = 'nus-wide'
    COCO = 'coco'

    PRE_TRAIN_NET_PATH = './data/imagenet-vgg-f.mat'
    PRE_TRAIN_MODEL = scio.loadmat(PRE_TRAIN_NET_PATH)

    def __init__(self, data_name, data_path, nbit=24, net_save_name='PatchH_net.pkl', batch_size=32, epoch=150, lr=0.1,
                 momentum=0, weight_decay=1e-5, lr_decay_epoch=50, lr_decay_rate=0.1, npatch=2, use_gpu=True):
        self.ETA = 2e-2
        self.CLIP = 1
        # save name for results of network
        self.NET_SAVE_NAME = net_save_name
        # parameters for network
        self.NBIT = nbit
        self.BATCH_SIZE = batch_size
        self.EPOCH = epoch
        self.LR = lr
        self.MOMENTUM = momentum
        self.WEIGHT_DECAY = weight_decay
        self.LR_DECAY_EPOCH = lr_decay_epoch
        self.LR_DECAY_RATE = lr_decay_rate
        # parameter for gpu
        self.USE_GPU = use_gpu
        # dataset set
        self.DATA_PATH = data_path

        self.NPATCH = npatch
        self.TEST_SIDE_LENGTH = 224
        self.TRAIN_SIDE_LENGTH = 192 + 32 * self.NPATCH
        self.train_transform = transforms.Compose([
            transforms.Resize([self.TRAIN_SIDE_LENGTH, self.TRAIN_SIDE_LENGTH], Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize([self.TEST_SIDE_LENGTH, self.TEST_SIDE_LENGTH], Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if type(data_name) != str:
            warnings.warn('Warning: data_name should be str type!')
        elif data_name.lower() == self.CIFAR10:  # cifar10 dataset
            self.DATA_NAME = self.CIFAR10
            # training set
            self.TRAIN_DATA = _CIFAR10(root=self.DATA_PATH, train=True, transform=self.train_transform)
            # retrieval set
            self.RETRI_DATA = _CIFAR10(root=self.DATA_PATH, train=True, transform=self.test_transform)
            # test set
            self.TEST_DATA = _CIFAR10(root=self.DATA_PATH, train=False, transform=self.test_transform)
            self.TEST_DATA2 = _CIFAR10(root=self.DATA_PATH, train=False, transform=self.train_transform)
            # parameters for dataset
            self.NUM_CA = 10  # the number of categories
            self.LABEL_IS_SINGLE = True  # whether the label is single data
            self.NUM_WORKERS = 1
            # parameters for dataset
            self.NUM_TR = self.TRAIN_DATA.__len__()  # the number of training set
            self.NUM_TE = self.TEST_DATA.__len__()  # the number of test set

            self.TRAIN_LABELS = torch.zeros((self.NUM_TR, 10))
            for i in range(self.NUM_TR):
                self.TRAIN_LABELS[i, self.TRAIN_DATA.train_labels[i]] = 1
            self.TEST_LABELS = torch.zeros((self.NUM_TE, 10))
            for i in range(self.NUM_TE):
                self.TEST_LABELS[i, self.TEST_DATA.test_labels[i]] = 1
        elif data_name.lower() == self.NUS:
            self.DATA_NAME = self.NUS
            # training set
            self.TRAIN_DATA = _NUS(root=self.DATA_PATH, train=True, transform=self.train_transform)
            # retrieval set
            self.RETRI_DATA = _NUS(root=self.DATA_PATH, train=True, transform=self.test_transform)
            # test set
            self.TEST_DATA = _NUS(root=self.DATA_PATH, train=False, transform=self.test_transform)
            self.TEST_DATA2 = _NUS(root=self.DATA_PATH, train=False, transform=self.train_transform)
            # parameters for dataset
            self.NUM_CA = 21  # the number of categories
            self.LABEL_IS_SINGLE = False  # whether the label is single data
            self.NUM_WORKERS = 4
            # parameters for dataset
            self.NUM_TR = self.TRAIN_DATA.__len__()  # the number of training set
            self.NUM_TE = self.TEST_DATA.__len__()  # the number of test set

            self.TRAIN_LABELS = torch.Tensor(self.TRAIN_DATA.train_labels)
            self.TEST_LABELS = torch.Tensor(self.TEST_DATA.test_labels)
        elif data_name.lower() == self.COCO:
            self.DATA_NAME = self.COCO
            # training set
            self.TRAIN_DATA = _COCO(root=self.DATA_PATH, train=True, transform=self.train_transform)
            # training set
            self.RETRI_DATA = _COCO(root=self.DATA_PATH, train=True, transform=self.test_transform)
            # test set
            self.TEST_DATA = _COCO(root=self.DATA_PATH, train=False, transform=self.test_transform)
            self.TEST_DATA2 = _COCO(root=self.DATA_PATH, train=False, transform=self.train_transform)
            # parameters for dataset
            self.NUM_CA = 91  # the number of categories
            self.LABEL_IS_SINGLE = False  # whether the label is single data
            self.NUM_WORKERS = 4
            # parameters for dataset
            self.NUM_TR = self.TRAIN_DATA.__len__()  # the number of training set
            self.NUM_TE = self.TEST_DATA.__len__()  # the number of test set

            self.TRAIN_LABELS = torch.Tensor(self.TRAIN_DATA.train_labels)
            self.TEST_LABELS = torch.Tensor(self.TEST_DATA.test_labels)

        # train loader
        self.TRAIN_LOADER = DataLoader(dataset=self.TRAIN_DATA, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)
        # retrieval loader
        self.RETRI_LOADER = DataLoader(dataset=self.RETRI_DATA, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.NUM_WORKERS)
        self.RETRI_LOADER2 = DataLoader(dataset=self.TRAIN_DATA, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.NUM_WORKERS)
        # test loader
        self.TEST_LOADER = DataLoader(dataset=self.TEST_DATA, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.NUM_WORKERS)
        self.TEST_LOADER2 = DataLoader(dataset=self.TEST_DATA2, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.NUM_WORKERS)
