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
            file_name = os.path.join(self.root_dir,
                                     self.rgbd_frame.iloc[idx, 0])
            rgb_h5, depth_h5 = self.load_h5(file_name)
            rgb_image = Image.fromarray(rgb_h5, mode='RGB')
            depth_image = Image.fromarray(depth_h5.astype('float32'), mode='F')
        elif self.input_format == 'png':
            rgb_name = os.path.join(self.root_dir,
                    # os.path.join("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_02/Depth-Anything_image_02",
                    # os.path.join("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_02/data/",
                    # os.path.join("D:\\dataset\\data\\2011_10_03_drive_0027_sync\\image_02\\Depth-Anything_image_02_grey",
                    #              self.rgbd_frame.iloc[idx, 0].split('/')[-1].split('.')[0]+'_depth.png'))
                                #  self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
                    os.path.join("D:\\dataset\\data\\FallingThings\\kitchen_0_result\\left_depth_uint8",
                                 self.rgbd_frame.iloc[idx, 0].replace('.jpg','.png')))
            # rgb_name = os.path.join(self.root_dir,
            #         os.path.join("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_center/image_02",
            #                      self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
            # with open(rgb_name, 'rb') as fRgb:
            rgb_image = Image.open(rgb_name)
            
            # depth_name = os.path.join(self.root_dir,
            #             os.path.join("/content/drive/MyDrive/Colab Notebooks/data/kitti/2011_10_03_drive_0027_sync/output_CREStereo",
            #                          self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
            # depth_name = os.path.join(self.root_dir,
            #             os.path.join("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/output_CREStereo_full",
            #                          self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
            depth_name = os.path.join(self.root_dir,
                        # os.path.join("D:\\dataset\\data\\2011_10_03_drive_0027_sync\\output_CREStereo_full",
                        #              self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
                        os.path.join("D:\\dataset\\data\\FallingThings\\kitchen_0_result\\output_CREStereo_full_200",
                                     self.rgbd_frame.iloc[idx, 0]))
            # with open(depth_name, 'rb') as fDepth:
            depth_image = Image.open(depth_name)

            # gt_name=os.path.join(self.root_dir,
            #             os.path.join("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_02/groundtruth",
            #                          self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
            # gt_name=os.path.join(self.root_dir,
            #             os.path.join("D:\\dataset\\data\\2011_10_03_drive_0027_sync\\image_02\\groundtruth_uint8_8",
            #                          self.rgbd_frame.iloc[idx, 0].split('/')[-1]))
            gt_name=os.path.join(self.root_dir,
                        os.path.join("D:\dataset\data\FallingThings\kitchen_0_result\left_gt_reverse_4_uint8_normalized",
                                     self.rgbd_frame.iloc[idx, 0].replace('.jpg','.depth.png')))
            gt_image=Image.open(gt_name)
        else:
            print('error: the input format is not supported now!')
            return None

        _s = np.random.uniform(1.0, 1.5)
        s = int(240*_s)
        degree = np.random.uniform(-5.0, 5.0)
        if self.split == 'train':
            print('val.py train')
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
            if np.random.uniform()<0.5:
                rgb_image = rgb_image.transpose(Image.FLIP_LEFT_RIGHT)
                depth_image = depth_image.transpose(Image.FLIP_LEFT_RIGHT)

            rgb_image = transforms.ToTensor()(rgb_image)
            if self.input_format == 'img':
                depth_image = transforms.ToTensor()(depth_image)
            else:
                depth_image = data_transform.ToTensor()(depth_image)
            depth_image = depth_image.div(_s)
            sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)
            sample = {'rgbd': rgbd_image, 'depth': depth_image}

        elif self.split == 'val':
            tRgb = data_transform.Compose([transforms.Resize(240),
                                           transforms.CenterCrop((228, 304)),
                                           transforms.ToTensor(),
                                        #    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        # transforms.Normalize((0.449,), (0.226,)),
                                           transforms.ToPILImage()])

            tDepth = data_transform.Compose([transforms.Resize(240),
                                             transforms.CenterCrop((228, 304))])
            gttDepth = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(240),
                # data_transform.ToTensorNormalize(),
                # transforms.ToPILImage(),
                # transforms.Resize(s),
                # data_transform.Rotation(degree),
                transforms.CenterCrop((228, 304))
                # transforms.CenterCrop((300 ,1000))
                # transforms.ToPILImage()
            ])           

            rgb_raw = tDepth(rgb_image)
            rgb_image = tRgb(rgb_image)
            depth_image = tDepth(depth_image)
            gt_image = gttDepth(gt_image)
            # rgb_image = transforms.ToTensor()(rgb_image)
            # rgb_raw = transforms.ToTensor()(rgb_raw)
            rgb_image = data_transform.ToTensor()(rgb_image)
            rgb_raw = data_transform.ToTensor()(rgb_raw)
            # if self.input_format == 'img':
            #     depth_image = transforms.ToTensor()(depth_image)
            #     gt_image = transforms.ToTensor()(gt_image)
            # else:
            depth_image = data_transform.ToTensor()(depth_image)
            gt_image = data_transform.ToTensor()(gt_image)
            # sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            sparse_image=depth_image
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)

            # sample = {'rgbd': rgbd_image, 'depth': depth_image, 'raw_rgb': rgb_raw }
            sample = {'rgbd': rgbd_image, 'depth': gt_image, 'raw_rgb': rgb_raw ,'old_depth':depth_image}

        return sample

    def createSparseDepthImage(self, depth_image, n_sample):
        # random_mask = torch.zeros(1, depth_image.size(1), depth_image.size(2))
        # n_pixels = depth_image.size(1) * depth_image.size(2)
        # n_valid_pixels = torch.sum(depth_image>0.0001)
        # perc_sample = n_sample/n_pixels
        # random_mask = torch.bernoulli(torch.ones_like(random_mask)*perc_sample)
        # sparse_depth = torch.mul(depth_image, random_mask)
        # return sparse_depth
        return depth_image

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
        rgb = f['rgb'][:].transpose(1,2,0)
        depth = f['depth'][:]
        return (rgb, depth)


