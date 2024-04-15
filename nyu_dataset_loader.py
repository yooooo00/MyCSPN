#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:07:52 2018

@author: norbot
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import data_transform
from PIL import Image, ImageOps
import h5py

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
imagenet_eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
imagenet_eigvec = np.array([[-0.5675,  0.7192,  0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948,  0.4203]], dtype=np.float32)


class NyuDepthDataset(Dataset):
    # nyu depth dataset 
    def __init__(self, csv_file, root_dir, split, n_sample=200, input_format = 'img'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rgbd_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.input_format = input_format
        self.n_sample = n_sample
    
    def __len__(self):
        return len(self.rgbd_frame)

    def __getitem__(self, idx):
        # read input image
        if self.input_format == 'img':
#            print('==> Input Format is image')
            rgb_name = os.path.join(self.root_dir,
                                    self.rgbd_frame.iloc[idx, 0])
            with open(rgb_name, 'rb') as fRgb:
                rgb_image = Image.open(rgb_name).convert('RGB')
            
            depth_name = os.path.join(self.root_dir,
                                      self.rgbd_frame.iloc[idx, 1])
            with open(depth_name, 'rb') as fDepth:
                depth_image = Image.open(depth_name)
                
        # read input hdf5
        elif self.input_format == 'hdf5':
#            print('==> Input Format is hdf5')
            file_name = os.path.join(self.root_dir,
                                     self.rgbd_frame.iloc[idx, 0])
            rgb_h5, depth_h5 = self.load_h5(file_name)   
#            print(depth_h5.dtype)
            rgb_image = Image.fromarray(rgb_h5, mode='RGB')
            depth_image = Image.fromarray(depth_h5.astype('float32'), mode='F')
#            plt.figure()
#            show_img(rgb_image)
#            plt.figure()
#            show_img(depth_image)
        elif self.input_format == 'png':
            rgb_name = os.path.join(self.root_dir,
                    os.path.join("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_02/Depth-Anything_image_02",
                                 self.rgbd_frame.iloc[idx, 0].split('/')[-1].split('.')[0]+'_depth.png'))
            # rgb_name = os.path.join(self.root_dir,
            #         os.path.join("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_center/image_02",
            #                      self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
            with open(rgb_name, 'rb') as fRgb:
                rgb_image = Image.open(rgb_name).convert('RGB')
            
            # depth_name = os.path.join(self.root_dir,
            #             os.path.join("/content/drive/MyDrive/Colab Notebooks/data/kitti/2011_10_03_drive_0027_sync/output_CREStereo",
            #                          self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
            depth_name = os.path.join(self.root_dir,
                        os.path.join("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/output_CREStereo_full",
                                     self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
            with open(depth_name, 'rb') as fDepth:
                depth_image = Image.open(depth_name).convert('L')

            gt_name=os.path.join(self.root_dir,
                        os.path.join("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_02/groundtruth",
                                     self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
            gt_image=Image.open(gt_name).convert('L')
            plt.imshow(gt_image)
            plt.show()
            # plt.imshow(depth_name)
            # plt.show()
            # plt.imshow(rgb_name)
            # plt.show()
        else:
            print('error: the input format is not supported now!')
            return None
        
        _s = np.random.uniform(1.0, 1.5)
        s = int(240*_s)
        degree = np.random.uniform(-5.0, 5.0)
        if self.split == 'train':
            tRgb = data_transform.Compose([transforms.Resize(s),
                                           data_transform.Rotation(degree),
                                           transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4),
#                                           data_transform.Lighting(0.1, imagenet_eigval, imagenet_eigvec)])
                                           transforms.CenterCrop((228, 304)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                           transforms.ToPILImage()])

            tDepth = data_transform.Compose([transforms.Resize(s),
                                             data_transform.Rotation(degree),
                                             transforms.CenterCrop((228, 304))])
            rgb_image = tRgb(rgb_image)
            depth_image = tDepth(depth_image)
            gt_image=tDepth(gt_image)
            if np.random.uniform()<0.5:
                rgb_image = rgb_image.transpose(Image.FLIP_LEFT_RIGHT)
                depth_image = depth_image.transpose(Image.FLIP_LEFT_RIGHT)
                gt_image = gt_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            rgb_image = transforms.ToTensor()(rgb_image)
            if self.input_format == 'img':
                depth_image = transforms.ToTensor()(depth_image)
                gt_image = transforms.ToTensor()(gt_image)
            else:
                depth_image = data_transform.ToTensor()(depth_image)
                gt_image = data_transform.ToTensor()(gt_image)
            depth_image = depth_image.div(_s)  
            gt_image = gt_image.div(_s)         
            # sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            sparse_image=depth_image
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)


        elif self.split == 'val':
            tRgb = data_transform.Compose([transforms.Resize(240),
                                           transforms.CenterCrop((228, 304)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                           transforms.ToPILImage()])

            tDepth = data_transform.Compose([transforms.Resize(240),
                                             transforms.CenterCrop((228, 304))])            
            rgb_image = tRgb(rgb_image)
            depth_image = tDepth(depth_image)
            gt_image = tDepth(gt_image)
            rgb_image = transforms.ToTensor()(rgb_image)
            if self.input_format == 'img':
                depth_image = transforms.ToTensor()(depth_image)
                gt_image = transforms.ToTensor()(gt_image)
            else:
                depth_image = data_transform.ToTensor()(depth_image)
                gt_image = data_transform.ToTensor()(gt_image)
            sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)
            
        # sample = {'rgbd': rgbd_image, 'depth': depth_image }
        sample = {'rgbd': rgbd_image, 'depth': gt_image }
        # plt.imshow(transforms.ToPILImage()(gt_image))
        # plt.imshow(transforms.ToPILImage()(rgb_image))
        # plt.imshow(transforms.ToPILImage()(sparse_image))
        # plt.show()
        return sample
    
    def createSparseDepthImage(self, depth_image, n_sample):
#         random_mask = torch.zeros(1, depth_image.size(1), depth_image.size(2))
#         n_pixels = depth_image.size(1) * depth_image.size(2)
#         n_valid_pixels = torch.sum(depth_image>0.0001)
# #        print('===> number of total pixels is: %d\n' % n_pixels)
# #        print('===> number of total valid pixels is: %d\n' % n_valid_pixels)
#         perc_sample = n_sample/n_pixels
#         random_mask = torch.bernoulli(torch.ones_like(random_mask)*perc_sample)
#         sparse_depth = torch.mul(depth_image, random_mask)

#         return sparse_depth
        return depth_image # 改为直接用整个depth_image 不再取点模拟
        

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
    #    print (f.keys())
        rgb = f['rgb'][:].transpose(1,2,0)
        depth = f['depth'][:]
        return (rgb, depth)


def show_img(image):
    """Show image"""
    plt.imshow(image)
    
def test_imgread():

    # train preprocessing   
#    nyudepth_dataset = NyuDepthDataset(csv_file='data/kitti_hdf5/kitti_hdf5_train.csv',
#                                       root_dir='.',
#                                       split = 'train',
#                                       n_sample = 200,
#                                       input_format='hdf5')
    nyudepth_dataset = NyuDepthDataset(csv_file='data/nyudepth_hdf5/nyudepth_hdf5_val.csv',
                                       root_dir='.',
                                       split = 'val',
                                       n_sample = 500,
                                       input_format='hdf5')    
    fig = plt.figure()
    for i in range(len(nyudepth_dataset)):
        sample = nyudepth_dataset[i]
        rgb = transforms.ToPILImage()(sample['rgbd'][0:3,:,:])
        depth = transforms.ToPILImage()(sample['depth'])
        sparse_depth = transforms.ToPILImage()(sample['rgbd'][3,:,:].unsqueeze(0))
        depth_mask = transforms.ToPILImage()(torch.sign(sample['depth']))
        sparse_depth_mask = transforms.ToPILImage()(sample['rgbd'][3,:,:].unsqueeze(0).sign())
        print(sample['rgbd'][0:3,:,:])
        invalid_depth = torch.sum(sample['rgbd'][3,:,:].unsqueeze(0).sign() < 0)
        print(invalid_depth)
#        print(sample['depth'].size())
#        print(torch.sign(sample['sparse_depth']))
        ax = plt.subplot(5, 4, i + 1)
        ax.axis('off')
        show_img(rgb)
        ax = plt.subplot(5, 4, i + 5)
        ax.axis('off')
        show_img(depth)
        ax = plt.subplot(5, 4, i + 9)
        ax.axis('off')
        show_img(depth_mask)
        ax = plt.subplot(5, 4, i + 13)
        ax.axis('off')
        show_img(sparse_depth)

        ax = plt.subplot(5, 4, i + 17)
        ax.axis('off')
        show_img(sparse_depth_mask)
        plt.imsave('sparse_depth.png', sparse_depth_mask)
        if i == 3:
            plt.show()
            break
    
#test_imgread()