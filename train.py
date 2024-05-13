#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 19:16:42 2018
@author: norbot
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse

from torch.autograd import Variable

import utils
import loss as my_loss
import lr_scheduler as lrs
from tqdm import tqdm

# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Sparse To Dense Training')

# net parameters
parser.add_argument('--n_sample', default=200, type=int, help='sampled sparse point number')
parser.add_argument('--data_set', default='nyudepth', type=str, help='train dataset')

# optimizer parameters
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (L2 penalty)')
parser.add_argument('--dampening', default=0.0, type=float, help='dampening for momentum')
parser.add_argument('--nesterov', '-n', action='store_true', help='enables Nesterov momentum')
parser.add_argument('--num_epoch', default=40, type=int, help='number of epoch for training')

# network parameters
parser.add_argument('--cspn_step', default=24, type=int, help='steps of propagation')
parser.add_argument('--cspn_norm_type', default='8sum', type=str, help='norm type of cspn')

# batch size
parser.add_argument('--batch_size_train', default=8, type=int, help='batch size for training')
parser.add_argument('--batch_size_eval', default=1, type=int, help='batch size for eval')

#data directory
parser.add_argument('--save_dir', default='result/base_line', type=str, help='result save directory')
parser.add_argument('--best_model_dir', default='result/base_line', type=str, help='best model load directory')
parser.add_argument('--train_list', default='data/nyudepth_hdf5/nyudepth_hdf5_train.csv', type=str, help='train data lists')
parser.add_argument('--eval_list', default='data/nyudepth_hdf5/nyudepth_hdf5_val.csv', type=str, help='eval data list')
parser.add_argument('--model', default='base_model', type=str, help='model for net')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--pretrain', '-p', action='store_true', help='load pretrained resnet model')
parser.add_argument('--resume_model_name', default='best_model.pth', type=str, help='resume model name')

args = parser.parse_args()

sys.path.append("./models")

import update_model
if args.model == 'cspn_unet':
    if args.data_set=='nyudepth':
        print("==> training model with cspn and unet on nyudepth")
        import torch_resnet_cspn_nyu as model
    elif args.data_set =='kitti':
        print("==> training model with cspn and unet on kitti")
        import torch_resnet_cspn_kitti as model
else:
    import torch_resnet as model

use_cuda = torch.cuda.is_available()


# global variable
best_rmse = sys.maxsize  # best test rmse
cspn_config = {'step': args.cspn_step, 'norm_type': args.cspn_norm_type}
start_epoch = 0 # start from epoch 0 or last checkpoint epoch
best_loss=sys.maxsize
best_loss_train=sys.maxsize

# Data
print('==> Preparing data..')
if args.data_set=='nyudepth':
    import nyu_dataset_loader as dataset_loader
    # trainset = dataset_loader.NyuDepthDataset(csv_file=args.train_list,
    #                                           root_dir='.',
    #                                           split = 'train',
    #                                           n_sample = args.n_sample,
    #                                           input_format='hdf5')
    # valset = dataset_loader.NyuDepthDataset(csv_file=args.eval_list,
    #                                         root_dir='.',
    #                                         split = 'val',
    #                                         n_sample = args.n_sample,
    #                                         input_format='hdf5')
    trainset = dataset_loader.NyuDepthDataset(csv_file=args.train_list,
                                              root_dir='/',
                                              split = 'train',
                                              n_sample = args.n_sample,
                                              input_format='png')
    valset = dataset_loader.NyuDepthDataset(csv_file=args.eval_list,
                                            root_dir='/',
                                            split = 'val',
                                            n_sample = args.n_sample,
                                            input_format='png')
# elif args.data_set =='kitti':
#     import kitti_dataset_loader as dataset_loader
#     trainset = dataset_loader.KittiDataset(csv_file=args.train_list,
#                                            root_dir='.',
#                                            split = 'train',
#                                            n_sample = args.n_sample,
#                                            input_format='hdf5')
#     valset = dataset_loader.KittiDataset(csv_file=args.eval_list,
#                                          root_dir='.',
#                                          split = 'val',
#                                          n_sample = args.n_sample,
#                                          input_format='hdf5')
# 
# else:
#     print("==> input unknow dataset..")

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size_train,
                                          shuffle=True,
                                          num_workers=0,
                                          pin_memory=True,
                                          drop_last=True)

if args.data_set=='nyudepth':
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batch_size_eval,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            drop_last=True)
# elif args.data_set == 'kitti':
#     valloader = torch.utils.data.DataLoader(valset,
#                                             batch_size=args.batch_size_eval,
#                                             shuffle=True,
#                                             num_workers=2,
#                                             pin_memory=True,
#                                             drop_last=True)
# Model

print('==> Prepare results folder and files...')
utils.log_file_folder_make_lr(args.save_dir)
print('==> Building model..')

if args.data_set == 'nyudepth':
    net = model.resnet18(pretrained = args.pretrain,
                         cspn_config=cspn_config)
# elif args.data_set == 'kitti':
#     net = model.resnet18(pretrained = args.pretrain,
#                          cspn_config=cspn_config)
# else:
#     print("==> input unknow dataset..")
import time
from datetime import datetime
if args.resume:
    # Load best model checkpoint.
    print('==> Resuming from best model..')
    best_model_path = os.path.join(args.best_model_dir, args.resume_model_name)
    print(best_model_path)
    assert os.path.isdir(args.best_model_dir), 'Error: no checkpoint directory found!'
    # best_model_dict = torch.load(best_model_path)
    # best_model_dict = update_model.remove_moudle(best_model_dict)
    # net.load_state_dict(update_model.update_model(net, best_model_dict))
    net.load_state_dict(torch.load(best_model_path),strict=False)
    # torch.cuda.empty_cache()
    # os.rename(best_model_path, os.path.join(args.best_model_dir, 'best_model_' + str(datetime.now().strftime('%Y%m%d_%H%M%S')) + '.pth'))
    # 怀疑有加载问题，尝试自己的
    # def remove_module_prefix(state_dict):
    #     """移除保存的模型键名中的'module.'前缀"""
    #     new_state_dict = {}
    #     for key in state_dict:
    #         if key.startswith('module.'):
    #             new_state_dict[key[len('module.'):]] = state_dict[key]
    #         else:
    #             new_state_dict[key] = state_dict[key]
    #     return new_state_dict
    # # 加载原始状态字典
    # original_state_dict = torch.load(best_model_path)
    # # 移除'module.'前缀
    # new_state_dict = remove_module_prefix(original_state_dict)
    # net.load_state_dict(new_state_dict)


if use_cuda:
    # torch.cuda.empty_cache()
    net.cuda()
    assert torch.cuda.device_count() == 1, 'only support single gpu'
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
# torch.cuda.empty_cache()

criterion = my_loss.Wighted_L1_Loss().cuda()
optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      nesterov=args.nesterov,
                      dampening=args.dampening)
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = lrs.ReduceLROnPlateau(optimizer, 'min') # set up scheduler


# Training
def train(epoch):
    net.train()
    total_step_train = 0
    train_loss = 0.0
    error_sum_train = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                       'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                       'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0,}

    tbar = tqdm(trainloader)
    for batch_idx, sample in enumerate(tbar):
        [inputs, targets] = [sample['rgbd'] , sample['depth']]
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        # 修改了loss
        sparse_depth = inputs.narrow(1,1,1).clone()
        rgb_image = inputs.narrow(1, 0, 1).clone()
        # refined_sparse_depth = net.depth_refinement_net(sparse_depth)
        refined_rgb_image=net.depth_refinement_net(rgb_image)
        # loss=criterion(outputs, targets)
        loss_outputs = criterion(outputs, targets)
        # loss_refined_depth= criterion.forward_depth(refined_sparse_depth, sparse_depth, targets)
        loss_refined_depth= criterion.forward_depth(refined_rgb_image, sparse_depth, targets)
        loss= loss_outputs + loss_refined_depth
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        loss_number=train_loss / (batch_idx + 1)
        error_str = 'Epoch: %d, loss=%.4f' % (epoch, train_loss / (batch_idx + 1))
        tbar.set_description(error_str)

        targets = targets.data.cpu()
        outputs = outputs.data.cpu()
        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)
        total_step_train += args.batch_size_train
        error_avg = utils.avg_error(error_sum_train,
                                    error_result,
                                    total_step_train,
                                    args.batch_size_train)
        # if batch_idx== 0:
        #     utils.print_error('training_result: step(average)',
        #                       epoch,
        #                       batch_idx,
        #                       loss,
        #                       error_result,
        #                       error_avg,
        #                       print_out=True)

    error_avg = utils.avg_error(error_sum_train,
                                error_result,
                                total_step_train,
                                args.batch_size_train)
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
    utils.log_result_lr(args.save_dir, error_avg, epoch, old_lr, False, 'train',loss_num=loss_number)

    # tmp_name = "epoch_%02d.pth" % (epoch)
    tmp_name="newest_model.pth"
    save_name = os.path.join(args.save_dir, tmp_name)
    # if epoch%100==0 and epoch!=0:
    torch.save(net.state_dict(), save_name)
    if epoch%20==19:
        torch.save(net.state_dict(), "epoch_%02d.pth" % (epoch))
    if loss_number<best_loss_train:
        best_loss_train = loss_number
        torch.save(net.state_dict(), "best_train_%.2f.pth"%(loss_number))


def val(epoch):
    global best_rmse,best_loss
    is_best_model = False
    net.eval()
    total_step_val = 0
    eval_loss = 0.0
    error_sum_val = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                     'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                     'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0,}

    tbar = tqdm(valloader)
    for batch_idx, sample in enumerate(tbar):
        [inputs, targets] = [sample['rgbd'] , sample['depth']]
        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs, volatile=True), Variable(targets)
                outputs = net(inputs)
        
        # 修改后的loss
        sparse_depth = inputs.narrow(1,1,1).clone()
        refined_sparse_depth = net.depth_refinement_net(sparse_depth)
        # loss = criterion(outputs, targets)
        loss_outputs = criterion(outputs, targets)
        loss_refined_depth= criterion.forward_depth(refined_sparse_depth, sparse_depth, targets)
        loss= loss_outputs + loss_refined_depth
        targets = targets.data.cpu()
        outputs = outputs.data.cpu()
        loss = loss.data.cpu()
        eval_loss += loss.item()
        loss_number=eval_loss / (batch_idx + 1)
        error_str = 'Epoch: %d, loss=%.4f' % (epoch, eval_loss / (batch_idx + 1))
        tbar.set_description(error_str)

        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)
        total_step_val += args.batch_size_eval
        error_avg = utils.avg_error(error_sum_val, error_result, total_step_val, args.batch_size_eval)

    utils.print_error('eval_result: step(average)',
                      epoch, batch_idx, loss,
                      error_result, error_avg, print_out=True)

    #log best_model
    # if utils.updata_best_model(error_avg, best_rmse):
    #     is_best_model = True
    #     best_rmse = error_avg['RMSE']
    if loss_number<best_loss:
        is_best_model = True
        best_loss = loss_number
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
    utils.log_result_lr(args.save_dir, error_avg, epoch, old_lr, is_best_model, 'eval',loss_num=loss_number)

    # saving best_model
    if is_best_model:
    # if :
        print('==> saving best model at epoch %d\n' % epoch)
        best_model_pytorch = os.path.join(args.save_dir, 'best_model.pth')
        torch.save(net.state_dict(), best_model_pytorch)

    #updata lr
    # scheduler.step(error_avg['MAE'], epoch)
    scheduler.step(loss_number, epoch)


def train_val():
    for epoch in range(0, args.num_epoch):
        train(epoch)
        val(epoch)

if __name__ == '__main__':
    train_val()
