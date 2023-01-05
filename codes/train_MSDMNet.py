# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import xlwt
import time
import datetime
import numpy as np

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from scipy.io import savemat
import cv2
import sys

from MSDMNet_model import MSDMNet
from metrics import get_metrics_reduced,get_metrics_full
from utils import PSH5Datasetfu, PSDataset, prepare_data, normlization, save_param, psnr_loss, ssim, save_img
from data import Data

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
model_str = 'MSDMNet'
satellite_str= r'Gaofen-1'
# . Get the parameters of your satellite
# sat_param = get_sat_param(satellite_str)
# if sat_param!=None:
#     ms_channels, pan_channels, scale = sat_param
# else:
#     print('You should specify `ms_channels`, `pan_channels` and `scale`! ')
ms_channels = 4
pan_channels = 1
scale = 4

# . Set the hyper-parameters for training
num_epochs = 200
lr = 0.0004
weight_decay = 0.0001
batch_size = 16
n_layer = 8
n_feat = 4
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
model_save_path = os.path.join(
    '../training/model/%s' % (model_str),
    timestamp + '_%s_layer' % (satellite_str))
save_path = os.path.join('../training/model/%s' % (model_str))
if not os.path.isdir(save_path):
    os.makedirs(save_path)

# . Get your model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)
net = MSDMNet(num_channels=n_feat ,batch_size=batch_size).to(device)
print(net)

# . Get your optimizer, scheduler and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
loss_fn = nn.L1Loss().to(device)

# 加载最优参'
if os.path.exists(save_path):
    net.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['net'],strict=False)
    print("load pretrained model successfully")

data_dir_ms= '/workspace/data/WorldView-4/MS_64'
data_dir_pan = '/workspace/data/WorldView-4/PAN_256'

trainloader = DataLoader(Data(data_dir_ms=data_dir_ms, data_dir_pan=data_dir_pan,phase='train'),
                         batch_size=batch_size,
                         shuffle=True,drop_last=True)

testloader = DataLoader(Data(data_dir_ms=data_dir_ms, data_dir_pan=data_dir_pan,phase='test'),
                        batch_size=1,drop_last=True)  # [N,C,H,W]
loader = {'train': trainloader,
          'validation': testloader}
print("param")
print('model:{}'.format(model_str)+'  satellite:{}' .format(satellite_str)+'  epoch:{}'.format(num_epochs)+
      '  lr:{}'.format(lr)+'  batch_size:{}'.format(batch_size)+'  n_feat:{}'.format( n_feat)+
      '  n_layer:{}'.format(n_layer))


'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

best_psnr_val, psnr_val, ssim_val = 0., 0., 0.
torch.backends.cudnn.benchmark = True
prev_time = time.time()
#
# net.load_state_dict(torch.load(os.path.join(save_path, 'last_best_net.pth'))['net'])
# optimizer.load_state_dict(torch.load(os.path.join(save_path, 'last_best_net.pth'))['optimizer'])

for epoch in range(num_epochs):
    ''' train '''
    theta=0.001
    for i, (lrms, pan, gt_2,gt_4, bms) in enumerate(loader['train']):
        # 0. preprocess data
        lrms, pan, gt_2,gt_4 , bms = lrms.cuda(), pan.cuda(),gt_2.cuda(), gt_4.cuda(), bms.cuda()

        # 1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        predHR,predhr_2,loss_hp= net(lrms, bms,pan)
        loss_4 = loss_fn(predHR, gt_4.detach())
        loss_2 = loss_fn(predhr_2, gt_2.detach())

        loss = (loss_4+loss_2+ theta * loss_hp).to(device)
        loss.backward()
        optimizer.step()

        # 2. print
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # 3. Log the ;scalar values
        print('[Epoch:{}/{}]'.format(epoch,num_epochs),end=" ")
        print('[Batch:{}/{}]'.format(i,len(loader['train'])),end=" ")
        print('[PSNR/Best: {:.5f}/{:.5f}]'.format(psnr_val,best_psnr_val),end=" ")
        print('[loss:{:.5f}]'.format(loss.item()),end=" ")
        print('[loss_2:{:.5f}]'.format(loss_2.item()), end=" ")
        print('[loss_4:{:.5f}]'.format(loss_4.item()), end=" ")
        print('[loss_hp:{:.5f}]'.format(loss_hp.item()), end=" ")
        print('[learning rate:{}]'.format(optimizer.state_dict()['param_groups'][0]['lr']),end=" ")
        print('[left_time:{}]'.format(time_left))


    ''' validation '''
    if epoch%10==0 or epoch==num_epochs:
        current_psnr_val = psnr_val

        psnr_val = 0.
        cc_val=0.
        sam_val=0.
        ergas_val=0.
        Q4_val = 0.

        metrics = torch.zeros(6, testloader.__len__())
        with torch.no_grad():
            net.eval()
            for i, (lrms, pan, gt_2, gt_4, bms) in enumerate(testloader):
                # 0. preprocess data
                lrms, pan, gt_2, gt_4, bms = lrms.cuda(), pan.cuda(), gt_2.cuda(), gt_4.cuda(), bms.cuda()
                predHR = net(lrms, bms,pan)
                metrics[:, i] = torch.Tensor(get_metrics_reduced(predHR, gt_4))
            psnr_val,cc_val,sam_val,ergas_val,Q4_val = metrics.mean(dim=1)
        print("test!!!!!!!!")
        print('[PSNR/test:{}]'.format( psnr_val),end=" ")
        print('[CC/test:{}]'.format(cc_val),end=" ")
        print('[SAM/test:{}]'.format(sam_val),end=" ")
        print('[ERGAS/test:{}]'.format(ergas_val),end=" ")
        print('[Q4_val/test:{}]'.format(Q4_val))


    ''' save model '''

    # Save the best weight
    if best_psnr_val < psnr_val:
        best_psnr_val = psnr_val
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        model_save_path = os.path.join(
            '../training/model/%s' % (model_str),
            timestamp + '_%s_layer' % (satellite_str))
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)
        torch.save({'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch},
                   os.path.join(model_save_path, 'best_net.pth'))
        torch.save({'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch},
                   os.path.join(save_path, 'last_best_net.pth'))
    scheduler.step()
    # ''' backtracking '''
    # if epoch > 0:
    #     if torch.isnan(loss):
    #         print(10 * '=' + 'Backtracking!' + 10 * '=')
    #         net.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['net'])
    #         optimizer.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['optimizer'])

'''
------------------------------------------------------------------------------
Test
------------------------------------------------------------------------------
'''

# 1. Load the best weight and create the dataloader for testing
net.load_state_dict(torch.load(os.path.join(save_path, 'last_best_net.pth'))['net'])
img_save_path=os.path.join('../test/%s/image' % (model_str))
if not os.path.isdir(img_save_path):
    os.makedirs(img_save_path)
# 2. Compute the metrics
metrics = torch.zeros(5, testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (lrms, pan, gt_2,gt_4 , bms) in enumerate(testloader):
        lrms, pan, gt_2,gt_4, bms = lrms.cuda(), pan.cuda(), gt_2.cuda(),gt_4.cuda(),bms.cuda()

        predHR = net(lrms, bms,pan)
        metrics[:, i] = torch.Tensor(get_metrics_reduced(predHR, gt_4))


list_PSNR = []
list_CC = []
list_SAM = []
list_ERGAS = []
list_Q4 = []
for n in range(testloader.__len__()):
    list_PSNR.append(metrics[0, n])
    list_CC.append(metrics[1, n])
    list_SAM.append(metrics[2, n])
    list_ERGAS.append(metrics[3, n])
    list_Q4.append((metrics[4,n]))

print("[list_psnr_mean:{}]".format( np.mean(list_PSNR)))
print("[list_cc_mean:{}]".format(np.mean(list_CC)))
print("[list_sam_mean:{}]".format(np.mean(list_SAM)))
print("[list_ergas_mean:{}]".format( np.mean(list_ERGAS)))
print("[list_Q4_mean:{}]".format(np.mean(list_Q4)))