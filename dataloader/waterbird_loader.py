import os, shutil
import sys
import time

import random
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms as transforms

import torch
from torch import nn
from torchvision.transforms import transforms
import pandas as pd



class WB_MultiDomainLoaderTriple(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, train_split, subsample=1, bd_aug=False):
        self.metadata_df = pd.read_csv(
            os.path.join(dataset_root_dir, 'metadata.csv'))
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.confounder_array = self.metadata_df['place'].values
        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.train_split = train_split
        self.dataset_root_dir = dataset_root_dir
        self.augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        # tr_example_path = os.path.join(dataset_root_dir, train_split[0])
        # self.categories_list = os.listdir(tr_example_path)

        # self.categories_list.sort()

        # self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        self.all_data, self.all_cate = self.make_dataset()


    def make_dataset(self):
        all_data=[]
        cnt=0

        for cnt in range(self.y_array.shape[0]):
            if self.split_array[cnt]==0:  # Train
                all_data.append([os.path.join(self.dataset_root_dir, self.filename_array[cnt]), self.y_array[cnt]])

        all_cate=[[], []]
        for d in all_data:
            each, id = d
            all_cate[id].append(each)

        return all_data, all_cate

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        img_path, id = self.all_data[index]
        img_x_ori = Image.open(img_path).convert("RGB")
        img_x = self.augment_transform(img_x_ori)
        img_x_2 = self.augment_transform(img_x_ori)
        img_x_3 = self.augment_transform(img_x_ori)
        img_x_4 = self.augment_transform(img_x_ori)

        return img_x, img_x_2, img_x_3, img_x_4


class WB_MultiDomainLoaderTripleFD(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, train_split, subsample=1, bd_aug=False):
        self.subsample = subsample
        self.train_split = train_split
        self.dataset_root_dir = dataset_root_dir

        self.metadata_df = pd.read_csv(
            os.path.join(dataset_root_dir, 'metadata.csv'))
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.confounder_array = self.metadata_df['place'].values
        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }


        self.augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.all_data, self.all_cate = self.make_dataset()


    def make_dataset(self):
        all_data = []
        cnt = 0

        flag=0
        if self.train_split=='val':
            flag=1
        elif self.train_split=='test':
            flag=2

        for cnt in range(self.y_array.shape[0]):
            if self.split_array[cnt] == flag:  # Train
                all_data.append([os.path.join(self.dataset_root_dir, self.filename_array[cnt]), self.y_array[cnt]])

        all_cate = [[], []]
        for d in all_data:
            each, id = d
            all_cate[id].append(each)


        return all_data, all_cate

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        img_path, label = self.all_data[index]
        img_x_ori = Image.open(img_path).convert("RGB")
        img_x = self.augment_transform(img_x_ori)

        img_xp_path = random.sample(self.all_cate[label], 1)[0]

        img_xp = Image.open(img_xp_path).convert("RGB")
        img_xp = self.transform(img_xp)

        return img_x,  img_xp, label



class WB_RandomData(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, all_split, sample_num=20):
        self.sample_num=sample_num
        self.all_split = all_split
        self.dataset_root_dir = dataset_root_dir

        self.metadata_df = pd.read_csv(
            os.path.join(dataset_root_dir, 'metadata.csv'))
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.confounder_array = self.metadata_df['place'].values
        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.all_data = self.make_dataset()
        self.index_list = [i for i in range(len(self.all_data))]

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



    def make_dataset(self):
        all_data = []
        cnt = 0
        for cnt in range(self.y_array.shape[0]):
            if self.split_array[cnt] == 0:  # Train
                all_data.append([os.path.join(self.dataset_root_dir, self.filename_array[cnt]), self.y_array[cnt]])

        return all_data

    def __getitem__(self, index):
        index_id = random.choice(self.index_list)
        img_path, label = self.all_data[index_id]
        img_x = Image.open(img_path).convert("RGB")
        img_x = self.transform(img_x)
        return img_x, label

    def __len__(self):
        return len(self.all_data)*self.sample_num

class WB_DomainTest(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, split, subsample=1, group=[0,0]):
        self.subsample = subsample
        self.split = split
        self.dataset_root_dir = dataset_root_dir

        self.metadata_df = pd.read_csv(
            os.path.join(dataset_root_dir, 'metadata.csv'))
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.confounder_array = self.metadata_df['place'].values
        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.group=group

        self.all_data = self.make_dataset()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def make_dataset(self):
        all_data = []
        cnt = 0
        if self.split=='val':
            flag=1
        elif self.split=='test':
            flag=2
        for cnt in range(self.y_array.shape[0]):
            if self.split_array[cnt] == flag:
                if self.y_array[cnt] == self.group[0] and self.confounder_array[cnt] == self.group[1]:
                    all_data.append([os.path.join(self.dataset_root_dir, self.filename_array[cnt]), self.y_array[cnt]])

        return all_data

    def __getitem__(self, index):
        img_path, label = self.all_data[index]
        img_x = Image.open(img_path).convert("RGB")
        img_x = self.transform(img_x)
        return img_x, label

    def __len__(self):
        return len(self.all_data)




class WB_DomainTest_Path(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, split, subsample=1, group=[0,0]):
        self.subsample = subsample
        self.split = split
        self.dataset_root_dir = dataset_root_dir

        self.metadata_df = pd.read_csv(
            os.path.join(dataset_root_dir, 'metadata.csv'))
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.confounder_array = self.metadata_df['place'].values
        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.group=group

        self.all_data = self.make_dataset()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def make_dataset(self):
        all_data = []
        cnt = 0
        if self.split=='val':
            flag=1
        elif self.split=='test':
            flag=2
        for cnt in range(self.y_array.shape[0]):
            if self.split_array[cnt] == flag:
                if self.y_array[cnt] == self.group[0] and self.confounder_array[cnt] == self.group[1]:
                    all_data.append([os.path.join(self.dataset_root_dir, self.filename_array[cnt]), self.y_array[cnt]])

        return all_data

    def __getitem__(self, index):
        img_path, label = self.all_data[index]
        img_x = Image.open(img_path).convert("RGB")
        img_x = self.transform(img_x)
        return img_x, label, img_path

    def __len__(self):
        return len(self.all_data)




