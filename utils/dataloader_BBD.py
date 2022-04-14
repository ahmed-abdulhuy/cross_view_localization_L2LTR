import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import scipy.io as sio

from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse
import pickle


# set BBD dataset for training
class TrainDataloader(Dataset):
    def __init__(self, args):
        
        self.polar = args.polar
        self.img_root = args.dataset_dir
        
        with open('../input/bbd-preprocessed/dataset.pkl', 'rb') as f:
            self.data_list = pickle.load(f)
        
        self.train_ratio = 0.9
        self.train_data_size = int(len(self.data_list.keys()) * self.train_ratio)


        self.transform = transforms.Compose(
            [transforms.Resize((args.img_size[0], args.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.transform_1 = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        #print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        # with open(self.train_list, 'r') as file:
        #     idx = 0
        #     for line in file:
        #         data = line.split(',')
        #         pano_id = (data[0].split('/')[-1]).split('.')[0]
        #         # satellite filename, streetview filename, pano_id
        #         if self.polar:
        #             item1 = self.img_root + data[0].replace('bing', 'polar').replace('jpg', 'png')
        #         else:
        #             item1 = self.img_root + data[0]

        #         item2 = self.img_root + data[1]

        #         self.id_list.append([item1, item2, pano_id])
        #         self.id_idx_list.append(idx)
        #         idx += 1
        
        idx = 0
        for sat_path in self.data_list:
            if idx == self.train_data_size:
                break
            grd_path = self.data_list[sat_path][0]
            self.id_list.append([sat_path, grd_path])
            self.id_idx_list.append(idx)
            idx +=1


        print('InputData::__init__: load BBD dataset', ' train_data_size =', self.train_data_size)

    def __getitem__(self, idx):

        
        x = Image.open(self.id_list[idx][1]).convert('RGB')
        x = self.transform(x)
        
        y = Image.open(self.id_list[idx][0]).convert('RGB')
        if self.polar:
            y = self.transform(y)
        else:
            y = self.transform_1(y)

        return x, y

    def __len__(self):
        return len(self.id_list)


class TestDataloader(Dataset):
    def __init__(self, args):
        print(args.polar)
        self.polar = args.polar

        self.img_root = args.dataset_dir

        with open('../input/bbd-preprocessed/dataset.pkl', 'rb') as f:
            self.data_list = pickle.load(f)

        self.test_ratio = 0.1
        self.test_data_size = int(len(self.data_list.keys()) * self.test_ratio)
        self.train_set_size = len(self.data_list.keys()) - self.test_data_size

        print('==========train_size: ', self.train_set_size, '===================')
        
        self.transform = transforms.Compose(
            [transforms.Resize((args.img_size[0], args.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.transform_1 = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        # with open(self.test_list, 'r') as file:
        #     idx = 0
        #     for line in file:
        #         data = line.split(',')
        #         pano_id = (data[0].split('/')[-1]).split('.')[0]
        #         # satellite filename, streetview filename, pano_id
        #         if self.polar:
        #             item1 = self.img_root + data[0].replace('bing', 'polar').replace('jpg', 'png')
        #         else:
        #             item1 = self.img_root + data[0]
                
        #         item2 = self.img_root + data[1]

        #         self.id_test_list.append([item1, item2, pano_id])
                
        #         self.id_test_idx_list.append(idx)
        #         idx += 1

        idx = 0
        for sat_path in self.data_list:
            if idx - self.train_set_size < 0:
                continue
            print('=============================we are in')

            grd_path = self.data_list[sat_path][0]
            self.id_list.append([sat_path, grd_path])
            self.id_test_idx_list.append(idx - self.train_set_size)
            idx +=1
        
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load BBD test Dataset', ' data_size =', self.test_data_size)


    def __getitem__(self, idx):
        
        x = Image.open(self.id_test_list[idx][1]).convert('RGB')
        
        x = self.transform(x)

        y = Image.open(self.id_test_list[idx][0]).convert('RGB')

        if self.polar:
            y = self.transform(y)
        else:
            y = self.transform_1(y)

        return x, y

    def __len__(self):
        return len(self.id_test_list)


# satalite data loader
class testSatDataloader(Dataset):
    def __init__(self, args):
        
        self.polar = args.polar
        self.img_root = args.dataset_dir

        with open('../input/bbd-preprocessed/dataset.pkl', 'rb') as f:
            self.train_list = pickle.load(f)

        self.transform = transforms.Compose(
            [transforms.Resize((args.img_size[0], args.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.transform_1 = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        #print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.sat_id_list = []
        # self.id_idx_list = []
        # with open(self.train_list, 'r') as file:
        #     idx = 0
        #     for line in file:
        #         data = line.split(',')
        #         pano_id = (data[0].split('/')[-1]).split('.')[0]
        #         # satellite filename, streetview filename, pano_id
        #         if self.polar:
        #             item1 = self.img_root + data[0].replace('bing', 'polar').replace('jpg', 'png')
        #         else:
        #             item1 = self.img_root + data[0]

        #         item2 = self.img_root + data[1]

        #         self.id_list.append([item1, item2, pano_id])
        #         self.id_idx_list.append(idx)
        #         idx += 1

        for sat_img in self.train_list:
            self.sat_id_list.append(sat_img)
            
        self.data_size = len(self.sat_id_list)

        # print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)

    def __getitem__(self, idx):

        
        y = Image.open(self.sat_id_list[idx]).convert('RGB')
        if self.polar:
            y = self.transform(y)
        else:
            y = self.transform_1(y)
        # print('*****************************************', np.array(x).shape)
        return y

    def __len__(self):
        return len(self.sat_id_list)


# ground data loader
class testGrdDataloader(Dataset):
    def __init__(self, args):
        
        self.polar = args.polar

        with open('../input/bbd-preprocessed/dataset.pkl', 'rb') as f:
            self.train_list = pickle.load(f)

        self.transform = transforms.Compose(
            [transforms.Resize((args.img_size[0], args.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.transform_1 = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        #print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.grd_id_list = []
        # with open(self.train_list, 'r') as file:
        #     idx = 0
        #     for line in file:
        #         data = line.split(',')
        #         pano_id = (data[0].split('/')[-1]).split('.')[0]
        #         # satellite filename, streetview filename, pano_id
        #         if self.polar:
        #             item1 = self.img_root + data[0].replace('bing', 'polar').replace('jpg', 'png')
        #         else:
        #             item1 = self.img_root + data[0]

        #         item2 = self.img_root + data[1]

        #         self.id_list.append([item1, item2, pano_id])
        #         self.id_idx_list.append(idx)
        #         idx += 1

        for grd_list in self.train_list.values():
            for grd_path in grd_list:
                self.grd_id_list.append(grd_path)
            
        self.data_size = len(self.grd_id_list)

        # print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)

    def __getitem__(self, idx):

        
        x = Image.open(self.grd_id_list[idx]).convert('RGB')
        x = self.transform(x)
        
        # print('*****************************************', np.array(x).shape)
        return x

    def __len__(self):
        return len(self.grd_id_list)
