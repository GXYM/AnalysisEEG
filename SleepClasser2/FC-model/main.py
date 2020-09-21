import os
import gc
import time
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
import errno

from dataload import TextSleep
from model import SleepModel
from config import config as cfg, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.shedule import FixLR
import matplotlib.pyplot as plt

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

lr = None
train_step = 0
accuracy_tests = list()
accuracy_trains = list()
accuracy_vals = list()


def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


def save_model(model, epoch, lr, optimzer, accuracy):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)
    save_path = os.path.join(save_dir, 'sleep_class_{}-{:.3f}.pth'.format(epoch, accuracy))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def train(model, train_loader, train_data, test_data, val_data, scheduler, optimizer, epoch):

    global train_step

    global accuracy_tests
    global accuracy_trains
    global accuracy_vals
    losses = AverageMeter(max=100)
    model.train()
    # scheduler.step()
    print('Epoch: {} : LR = {}'.format(epoch, scheduler.get_lr()))
    for i, data in enumerate(train_loader):
        train_step += 1
        data = to_device(data)
        if data.shape[0] != cfg.batch_size:
            continue
        output = model(data[:, 1:])
        target = data[:, 0].long()
        loss = F.nll_loss(output, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        losses.update(loss.item())
        gc.collect()

        if i % cfg.display_freq == 0:
            print("({:d} / {:d}), loss: {:.3f}".format(i, len(train_loader), loss.item()))

    if epoch % cfg.save_freq == 0:
        labels_test = test_data[:, 0].long()
        output_test = model(test_data[:, 1:])
        pred_test = output_test.data.max(1, keepdim=True)[1]
        correct_test = pred_test.eq(labels_test.data.view_as(pred_test)).cpu().sum()
        accuracy_test = correct_test*100.0/labels_test.shape[0]
        accuracy_tests.append(round(accuracy_test.item(), 3))

        labels_train = train_data[:, 0].long()
        output_train = model(train_data[:, 1:])
        pred_train = output_train.data.max(1, keepdim=True)[1]
        correct_train = pred_train.eq(labels_train.data.view_as(pred_train)).cpu().sum()
        accuracy_train = correct_train * 100.0 / labels_train.shape[0]
        accuracy_trains.append(round(accuracy_train.item(), 3))

        labels_val = val_data[:, 0].long()
        output_val = model(val_data[:, 1:])
        pred_val = output_val.data.max(1, keepdim=True)[1]
        correct_val = pred_val.eq(labels_val.data.view_as(pred_val)).cpu().sum()
        accuracy_val = correct_val * 100.0 / labels_val.shape[0]
        accuracy_vals.append(round(accuracy_val.item(), 3))
        print("accuracy_train: {}; accuracy_val: {}; accuracy_test: {}"
              .format(accuracy_train, accuracy_val, accuracy_test))

    # if epoch % cfg.save_freq == 0:
    #     save_model(model, epoch, scheduler.get_lr(), optimizer)

    print('Training Loss: {}'.format(losses.avg))


def main():

    global lr
    all_data = TextSleep('.', is_training=True)
    test_data = torch.from_numpy(all_data.test_data).float()
    test_data = to_device(test_data)

    train_data = torch.from_numpy(all_data.train_data).float()
    train_data = to_device(train_data)

    val_data = torch.from_numpy(all_data.val_data).float()
    val_data = to_device(val_data)

    R = all_data.R

    train_loader = data.DataLoader(all_data, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    # Model
    model = SleepModel(5, is_training=True)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, cfg.resume)
    
    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)

    print('Start training sleep model.')
    for epoch in range(cfg.start_epoch, cfg.start_epoch + cfg.max_epoch+1):
        scheduler.step()
        train(model, train_loader, train_data, test_data, val_data, scheduler, optimizer, epoch)

    with open("./result/{}.txt".format(R), "w") as f:
        str_train = ','.join([str(i) for i in accuracy_trains])
        str_test = ','.join([str(i) for i in accuracy_tests])
        str_val = ','.join([str(i) for i in accuracy_vals])
        f.write("{}\n{}\n{}".format(str_train, str_test, str_val))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2020)
    torch.manual_seed(2020)
    # parse arguments
    print_config(cfg)

    # main
    main()

