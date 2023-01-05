import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import xlwt
import pdb
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from scipy.io import savemat
import cv2
import sys

sys.path.append("/home/zhangguiwei/KengKeng/1")
from MPCNN_model import MPCNN
from metrics import get_metrics_reduced,get_metrics_full
from utils import PSH5Datasetfu, PSDataset, prepare_data, normlization, save_param, psnr_loss, ssim,save_img
from data import Data

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

model_str = 'MPCNN-atten-newunet'
satellite_str= r'WorldView4'

ms_channels = 4
pan_channels = 1
scale = 4
n_feat=4
batch_size=16

# . Get your model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)
net = MPCNN(num_channels=n_feat,batch_size=batch_size ).to(device)

data_dir_ms= '/workspace/data/Gaofen1/MS_64'
data_dir_pan = '/workspace/data/Gaofen1/PAN_256'

testloader = DataLoader(Data(data_dir_ms=data_dir_ms, data_dir_pan=data_dir_pan,phase='test'),
                        batch_size=1)  # [N,C,H,W]


loader = {'validation': testloader}

# print("param")
# print('model:{}'.format(model_str)+'  satellite:{}' .format(satellite_str)+'  epoch:{}'.format(num_epochs)+
#       '  lr:{}'.format(lr)+'  batch_size:{}'.format(batch_size)+'  n_feat:{}'.format( n_feat)+
#       '  n_layer:{}'.format(n_layer))

# model_save_path = os.path.join(
#     '../training/model/%s' % (model_str),
#     timestamp + '_%s_layer' % (satellite_str))
save_path = os.path.join('../training/model/%s' % (model_str))
if not os.path.isdir(save_path):
    os.makedirs(save_path)

#
# 1. Load the best weight and create the dataloader for testing
net.load_state_dict(torch.load(os.path.join(save_path, 'last_best_net.pth'),map_location='cuda:0')['net'])
img_save_path=os.path.join('../test/%s/image' % (model_str))
memory_img_save_path=os.path.join('../test/%s/memory' % (model_str))
highpass_img_save_path=os.path.join('../test/%s/high_pass' % (model_str))
visual_img_save_path=os.path.join('../test/%s/visual' % (model_str))
if not os.path.isdir(img_save_path):
    os.makedirs(img_save_path)

import csv
f = open('./result.csv','w',newline='')
writer  =  csv.writer(f)
# 2. Compute the metrics
metrics = torch.zeros(6, testloader.__len__())
metrics_full = torch.zeros(3, testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (name,lrms, pan,gt_2,gt_4 , bms) in enumerate(testloader):
        lrms, pan,gt_2,gt_4, bms = lrms.cuda(), pan.cuda(),gt_2.cuda(),gt_4.cuda(),bms.cuda()
        # lrms,_ = normlization(lrms.cuda())
        # pan,_ = normlization(pan.cuda())
        # gt,_ = normlization(gt.cuda())

        predHR,d2_,dd2_= net(lrms, bms ,pan)
        if not os.path.isdir(os.path.join("../observe/{}".format(i))):
            os.mkdir(os.path.join("../observe/{}".format(i)))
        import skimage.io as io
        io.imsave(os.path.join('../observe/{}'.format(i), "d2_.png"), ((d2_[0]-d2_.min())/(d2_.max()-d2_.min())*255).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy())
        io.imsave(os.path.join('../observe/{}'.format(i), "dd2_.png"), ((dd2_[0]-dd2_.min())/(dd2_.max()-dd2_.min())*255).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy())
        io.imsave(os.path.join("../observe/{}".format(i), 'gtRGB.png'),
                  (gt_4[0][:3,:,:].clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()))
        io.imsave(os.path.join("../observe/{}".format(i), 'pan.png'),
                  (pan[0].clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()))
        metrics[:, i] = torch.Tensor(get_metrics_reduced(predHR, gt_4))
        metrics_full[:, i] = torch.Tensor(get_metrics_full(predHR/255,lrms/255,pan/255))
        # import kornia
        # pred_y = kornia.color.rgb_to_ycbcr(predHR[:,0:3,:,:])[:,0:1,:,:]
        # gt_y = kornia.color.rgb_to_ycbcr(gt_4[:, 0:3, :, :])[:,0:1,:,:]
        # pnsr = psnr_loss(pred_y,gt_y,max_val=1.)
        print(metrics[:, i])
        print(metrics_full[:,i])
        # writer.writerow([name,metrics[0,i],metrics[1,i],metrics[2,i],metrics[3,i],metrics[4,i],metrics[5,i],metrics_full[0,i],metrics_full[1,i],metrics_full[2,i]])
        # metrics[0,i]=pnsr
        # print(metrics[:,i])

        # for i,memory in enumerate(net.Memory_4.units.weight.data):
        #     memory=memory.view(1,4,64,64)
        #     save_feature_img('../memory','memory_{}.tif'.format(i),memory)

        # # save_feature_img("../visual-c/{}".format(i),'s1.tif',s1)
        # save_feature_img("../visual-64/{}".format(i), 'm1.tif', m1)
        # # save_feature_img("../visual-c/{}".format(i), 's1_.tif', s1_)
        # # save_feature_img("../visual-c/{}".format(i),'s2.tif',s2)
        # save_feature_img("../visual-64/{}".format(i), 'm2.tif', m2)
        # # save_feature_img("../visual-c/{}".format(i), 's2_.tif', s2_)
        # save_feature_img("../visual-64/{}".format(i), 'gt.tif', gt_4)
        # save_feature_img("../visual-64/{}".format(i), 'bms.tif', bms)
        # save_feature_img("../visual-64/{}".format(i), 'predHR.tif', predHR)
        # save_feature_img("../visual-64/{}".format(i), 'hp_pan_2.tif', hp_pan_2)
        # save_feature_img("../visual-64/{}".format(i), 'hp_pan_4.tif', hp_pan_4)
        # save_img("../visual-64/{}".format(i), 'gt_origin.tif', gt_4)
        # save_img("../visual-64/{}".format(i), 'pan_origin.tif', pan)
        # save_img("../visual-64/{}".format(i), 'predHR_origin.tif', predHR)


        # import pdb
        # pdb.set_trace()
list_PSNR = []
list_SSIM = []
list_CC = []
list_SAM = []
list_ERGAS = []
list_Q4 = []

list_DS = []
list_Dlambda = []
list_QNR= []
for n in range(testloader.__len__()):
    list_PSNR.append(metrics[0, n])
    list_SSIM.append(metrics[1, n])
    list_CC.append(metrics[2, n])
    list_SAM.append(metrics[3, n])
    list_ERGAS.append(metrics[4, n])
    list_Q4.append(metrics[5,n])
    list_QNR.append(metrics_full[0,n])
    list_DS.append(metrics_full[1, n])
    list_Dlambda.append(metrics_full[2, n])

print("[list_psnr_mean:{}]".format( np.mean(list_PSNR)))
print("[list_ssim_mean:{}]".format( np.mean(list_SSIM)))
print("[list_cc_mean:{}]".format(np.mean(list_CC)))
print("[list_sam_mean:{}]".format(np.mean(list_SAM)))
print("[list_ergas_mean:{}]".format( np.mean(list_ERGAS)))
print("[list_Q4_mean:{}]".format(np.mean(list_Q4)))
print("[list_QNR_mean:{}]".format(np.mean(list_QNR)))
print("[list_DS_mean:{}]".format(np.mean(list_DS)))
print("[list_Dlambda_mean:{}]".format(np.mean(list_Dlambda)))