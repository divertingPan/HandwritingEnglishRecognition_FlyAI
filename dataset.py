#!/usr/bin/python
# encoding: utf-8

import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler
import cv2
from character import cut_character
from path import DATA_PATH, DATA_ID
import matplotlib.pyplot as plt


class OCRDataset(Dataset):

    def __init__(self, img_path=None, label_value=None, mode='train', transform=None, target_transform=None):
        self.img_path = img_path
        self.label_value = label_value
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        if self.mode == 'train':
            try:
                # img = cv2.imread(os.path.join(DATA_PATH, DATA_ID, self.img_path[index]))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cut_character(os.path.join(DATA_PATH, DATA_ID, self.img_path[index]))

                img = Image.fromarray(np.uint8(img))

            except:
                # print('Cannot get signature area: %s' % self.img_path[index])
                return self[(index + 1) % len(self.img_path)]
        else:
            try:
                # img = cv2.imread(image_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cut_character(os.path.join(DATA_PATH, DATA_ID, self.img_path[index]))
                img = Image.fromarray(np.uint8(img))
            except:
                img = cv2.imread(os.path.join(DATA_PATH, DATA_ID, self.img_path[index]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = Image.fromarray(np.uint8(img))

        label = self.label_value[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
