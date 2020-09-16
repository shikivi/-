import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset

class ToTensor(object):
    def __call__(self, sample):
        result = {}
        for k in sample:
            if k == 'rect':
                result[k] = torch.IntTensor(sample[k])
            else:
                result[k] = torch.FloatTensor(sample[k])
        return result


class InpaintingDataset(Dataset):
    def __init__(self, info_list, root_dir='', image_size=(256, 256), transform=None):
        self.filenames = open(info_list, 'rt').read().splitlines()
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        np.random.seed(2020)

    def __len__(self):
        return len(self.filenames)

    def read_image(self, filepath):

        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        if h != self.image_size[0] or w != self.image_size[1]:
            ratio = max(1.0*self.image_size[0]/h, 1.0*self.image_size[1]/w)
            res_image = cv2.resize(image, None, fx=ratio, fy=ratio)
            h, w, _ = res_image.shape
            h_idx = (h-self.image_size[0]) // 2
            w_idx = (w-self.image_size[1]) // 2
            res_image = res_image[h_idx:h_idx+self.image_size[0], w_idx:w_idx+self.image_size[1],:]
            res_image = np.transpose(res_image, [2, 0, 1])
        else:
            res_image = np.transpose(image, [2, 0, 1])
        return res_image

    def __getitem__(self, idx):
        image = self.read_image(os.path.join(self.root_dir, self.filenames[idx]))
        sample = {'gt': image}
        if self.transform:
            sample = self.transform(sample)
        return sample
