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
        
        with open('../input/reduced-bdd-one-per-trajictory/dataset.pkl', 'rb') as f:
            self.dataset = pickle.load(f)
            self.data_list = self.dataset['train_data']
        
        if self.polar:
            self.tmp_data_list = self.data_list.copy()
            new_dir = '/kaggle/input/aerials-polar/aerials'
            for sat_path in self.tmp_data_list:
                self.data_list[new_dir + sat_path.split('aerials')[-1]] = self.data_list.pop(sat_path)
        
        # self.train_ratio = 0.9
        # self.train_data_size = int(len(self.data_list) * self.train_ratio)
        self.train_data_size = len(self.data_list)


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
        
        idx = 0
        for sat_path in self.data_list:
            # if idx == self.train_data_size:
            #     break
            for grd_path in self.data_list[sat_path]:
                # grd_path = self.data_list[sat_path][2]
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
    def __init__(self, args, dataset_index):

        with open('/kaggle/input/divide-bbd-to-5-datasets/bbd_test_distributed.pkl', 'rb') as f:
            self.dataset = pickle.load(f)
            self.data_list = self.dataset[dataset_index]
        

        self.test_data_size = len(self.data_list)
        print(f"{'='*30}Dataset Index:{dataset_index}{'='*30}")
        print(f"{'='*30}Test_data Size:{self.test_data_size}{'='*30}")

        
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

        idx = 0
        for sat_path in self.data_list:
            grd_path = self.data_list[sat_path]
            self.id_test_list.append([sat_path, grd_path])
            self.id_test_idx_list.append(idx)
            idx +=1

        
        self.test_data_size = len(self.id_test_list)


    def __getitem__(self, idx):
        
        x = Image.open(self.id_test_list[idx][1]).convert('RGB')
        
        x = self.transform(x)

        y = Image.open(self.id_test_list[idx][0]).convert('RGB')

        y = self.transform_1(y)

        return x, y

    def __len__(self):
        return len(self.id_test_list)


# satalite data loader
class satDataloader(Dataset):
    def __init__(self, args):
        
        self.polar = args.polar

        with open('../input/bbd-train-and-val/dataset.pkl', 'rb') as f:
            self.data_list = pickle.load(f)
        
        if self.polar:
            self.tmp_data_list = self.data_list.copy()
            new_dir = '/kaggle/input/aerials-polar/aerials'
            for sat_path in self.tmp_data_list:
                self.data_list[new_dir + sat_path.split('aerials')[-1]] = self.data_list.pop(sat_path)

        self.transform = transforms.Compose(
            [transforms.Resize((args.img_size[0], args.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.transform_1 = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        print('InputData::__init__: load from %s' % new_dir)
        self.__cur_id = 0  # for training
        self.sat_id_list = []

        for sat_img in self.data_list:
            self.sat_id_list.append(sat_img)
            
        self.data_size = len(self.sat_id_list)

        print('InputData::__init__: load', new_dir, ' data_size =', self.data_size)

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
class grdDataloader(Dataset):
    def __init__(self, args):
        
        self.polar = args.polar

        with open('../input/bbd-train-and-val/dataset.pkl', 'rb') as f:
            self.data_list = pickle.load(f)
        
        if self.polar:
            self.tmp_data_list = self.data_list.copy()
            new_dir = '/kaggle/input/aerials-polar/aerials'
            for sat_path in self.tmp_data_list:
                self.data_list[new_dir + sat_path.split('aerials')[-1]] = self.data_list.pop(sat_path)

        self.transform = transforms.Compose(
            [transforms.Resize((args.img_size[0], args.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.transform_1 = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        print('InputData::__init__: load from %s' % new_dir)
        self.__cur_id = 0  # for training
        self.grd_id_list = []

        for grd_list in self.data_list.values():
            for grd_path in grd_list:
                self.grd_id_list.append(grd_path)
            
        self.data_size = len(self.grd_id_list)

        print('InputData::__init__: load', new_dir, ' data_size =', self.data_size)

    def __getitem__(self, idx):

        
        x = Image.open(self.grd_id_list[idx]).convert('RGB')
        x = self.transform(x)
        
        # print('*****************************************', np.array(x).shape)
        return x

    def __len__(self):
        return len(self.grd_id_list)
