from easydict import EasyDict
import torch
import os

config = EasyDict()

config.gpu = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

# dataloader jobs number
config.num_workers = 0

# batch_size
config.batch_size = 64

# training epoch number
config.max_epoch = 1000

config.start_epoch = 0

# learning rate
config.lr = 1e-2
config.momentum = 0.9
config.optim = "SGD"

# using GPU
config.cuda = True
config.resume = None
config.display_freq = 5
config.save_freq = 1
config.save_dir = "./model"
config.exp_name = "sleep1"

config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
