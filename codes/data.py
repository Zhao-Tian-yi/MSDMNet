#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-23 14:57:22
LastEditTime: 2021-01-19 20:57:29
@Description: file content
'''
import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])


def load_img(filepath):
    img = Image.open(filepath)

    return img

def transform():
    return Compose([
        ToTensor(),
    ])

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(ms_image, lrms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = lrms_image.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    lrms_image = lrms_image.crop((iy, ix, iy + ip, ix + ip))
    ms_image = ms_image.crop((ty, tx, ty + tp, tx + tp))
    pan_image = pan_image.crop((ty, tx, ty + tp, tx + tp))
    bms_image = bms_image.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lrms_image, pan_image, bms_image, info_patch


def augment(ms_image, lrms_image, pan_image, bms_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        ms_image = ImageOps.flip(ms_image)
        lrms_image = ImageOps.flip(lrms_image)
        pan_image = ImageOps.flip(pan_image)
        # bms_image = ImageOps.flip(bms_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            ms_image = ImageOps.mirror(ms_image)
            lrms_image = ImageOps.mirror(lrms_image)
            pan_image = ImageOps.mirror(pan_image)
            # bms_image = ImageOps.mirror(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            ms_image = ms_image.rotate(180)
            lrms_image = lrms_image.rotate(180)
            pan_image = pan_image.rotate(180)
            # bms_image = pan_image.rotate(180)
            info_aug['trans'] = True

    return ms_image, lrms_image, pan_image, info_aug


class Data(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan,phase, transform=transform(), upscale = 4):
        super(Data, self).__init__()
        self.phase=phase
        if phase == "train":
            self.ms_image_filenames = [join(data_dir_ms+'_train', x) for x in listdir(data_dir_ms+'_train') if is_image_file(x)]
            self.pan_image_filenames =[join(data_dir_pan+'_train', x) for x in listdir(data_dir_pan+'_train') if is_image_file(x)]
        else:
            self.ms_image_filenames = [join(data_dir_ms + '_test', x) for x in listdir(data_dir_ms + '_test') if
                                       is_image_file(x)]
            self.pan_image_filenames = [join(data_dir_pan + '_test', x) for x in listdir(data_dir_pan + '_test') if
                                        is_image_file(x)]

        # self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = upscale
        self.transform = transform
        # self.data_augmentation = cfg['data']['data_augmentation']
        # self.normalize = cfg['data']['normalize']
        # self.cfg = cfg

    def __getitem__(self, index):


        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])

        _, file = os.path.split(self.ms_image_filenames[index])
        # Wald's协议
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor,
                                  ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        ms_image_2 = ms_image.resize(
            (int(ms_image.size[0] / self.upscale_factor*2), int(ms_image.size[1] / self.upscale_factor*2)), Image.BICUBIC)

        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                    pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        lrms_image = ms_image.resize(
            (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)), Image.BICUBIC)
        bms_image = rescale_img(lrms_image, self.upscale_factor)
        lrpan_image=pan_image.resize(
            (int(pan_image.size[0] / self.upscale_factor), int(pan_image.size[1] / self.upscale_factor)), Image.BICUBIC)
        # ms_image, lrms_image, pan_image, bms_image, _ = get_patch(ms_image, lrms_image, pan_image, bms_image,
        #                                                          self.patch_size, scale=self.upscale_factor)
        # if self.data_augmentation:
        #     ms_image, lrms_image, pan_image, _ = augment(ms_image, lrms_image, pan_image)

        if self.transform:
            ms_image = self.transform(ms_image)*255
            lrms_image = self.transform(lrms_image)*255
            lrpan_image = self.transform(lrpan_image)*255
            bms_image = self.transform(bms_image)*255
            ms_image_2 = self.transform(ms_image_2)*255
        return lrms_image, lrpan_image,ms_image_2, ms_image,bms_image

    def __len__(self):
        return len(self.ms_image_filenames)


class Data_test(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, transform=transform(), upscale = 4):
        super(Data_test, self).__init__()
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.upscale_factor = upscale
        self.transform = transform
        # self.data_augmentation = cfg['data']['data_augmentation']
        # self.normalize = cfg['data']['normalize']
        # self.cfg = cfg

    def __getitem__(self, index):

        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor,
                                  ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lrms_image = ms_image.resize(
            (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                    pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        # bms_image = rescale_img(lrms_image, self.upscale_factor)

        # if self.data_augmentation:
        #     ms_image, lrms_image, pan_image, _ = augment(ms_image, lrms_image, pan_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            lrms_image = self.transform(lrms_image)
            pan_image = self.transform(pan_image)
            # bms_image = self.transform(bms_image)

        # if self.normalize:
        #     ms_image = ms_image * 2 - 1
        #     lrms_image = lrms_image * 2 - 1
        #     pan_image = pan_image * 2 - 1
        #     bms_image = bms_image * 2 - 1

        return  lrms_image, pan_image,ms_image

    def __len__(self):
        return len(self.ms_image_filenames)


# class Data_eval(data.Dataset):
#     def __init__(self, image_dir, upscale_factor, cfg, transform=None):
#         super(Data_eval, self).__init__()
#
#         self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
#         self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
#
#         self.upscale_factor = cfg['data']['upsacle']
#         self.transform = transform
#         self.data_augmentation = cfg['data']['data_augmentation']
#         # self.normalize = cfg['data']['normalize']
#         self.cfg = cfg
#
#     def __getitem__(self, index):
#
#         lrms_image = load_img(self.ms_image_filenames[index])
#         pan_image = load_img(self.pan_image_filenames[index])
#         _, file = os.path.split(self.ms_image_filenames[index])
#         lrms_image = lrms_image.crop((0, 0, lrms_image.size[0] // self.upscale_factor * self.upscale_factor,
#                                    lrms_image.size[1] // self.upscale_factor * self.upscale_factor))
#         pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
#                                     pan_image.size[1] // self.upscale_factor * self.upscale_factor))
#         # bms_image = rescale_img(lrms_image, self.upscale_factor)
#
#         if self.data_augmentation:
#             lrms_image, pan_image, bms_image, _ = augment(lrms_image, pan_image, bms_image)
#
#         if self.transform:
#             lrms_image = self.transform(lrms_image)
#             pan_image = self.transform(pan_image)
#             # bms_image = self.transform(bms_image)
#
#         # if self.normalize:
#         #     lrms_image = lrms_image * 2 - 1
#         #     pan_image = pan_image * 2 - 1
#         #     bms_image = bms_image * 2 - 1
#
#         return lrms_image, pan_image, file
#
#     def __len__(self):
#         return len(self.ms_image_filenames)