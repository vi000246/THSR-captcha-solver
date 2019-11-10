# -*- coding: utf-8 -*-
"""Dataset module for training

This module include Dataset class and some utils for training.

"""
import torch
import os
import random
import numpy as np
from torchvision import transforms as T
from torch.utils import data
from pre_process import pre_process
from PIL import Image


def table(c):
    """Tranfer char to index.

    Args:
        c (str): one char.

    Returns:
        int: index of c.
    """
    i = ord(c)
    if i < 65:
        i -= 48
    else:
        i -= 55
    return i

def transforms(img_dir, img_name, pre=False):
    """transforms the img.

    Args:
        img_dir (str): dir. of image.
        img_name (str): file name of image.
        pre (bool): toggle pre-process

    Returns:
        tensor: image after transforms.
    """
    if pre:
        cache_path = os.path.join(img_dir, 'cache', img_name+'pre_1.png')
        if os.path.isfile(cache_path):
            data = Image.open(cache_path).convert('L')
        else:
            data = pre_process(os.path.join(img_dir, img_name), remove_curve=True)
            data.save(cache_path)
    else:
        data = Image.open(os.path.join(img_dir, img_name)).convert('L')
    transforms = T.Compose([
        T.Resize((128,128)),
        T.ToTensor(),
    ]) 
    data = transforms(data)
    return data

class Data(data.Dataset):
    def __init__(self, train, pre_process=False, dir='../captcha'):
        """__init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Note:
            Please modify slice size at line 80 and 83 to meet train size and val. size you want.

        Args:
            train (bool): load training dataset or not.
            pre_process (bool): enable pre_process or not
            dir (str): dataset dir.

        """
        self.dir = dir
        self.png_dict = self.load_csv()
        self.png_list = tuple(self.png_dict.keys())
        self.pre_process = pre_process
        if train:
            self.png_list = self.png_list[:30000] # modify here to change train set size
        else:
            self.png_list = self.png_list[30000:31000] # modify here to change val. set size
        # random.shuffle(self.png_list)


    def load_csv(self, name='label.csv'):
        """load label csv.

        Args:
            name( str ): CSV file name.

        Returns:
            dict: dict of image file names and labels.

        """
        png_dict = {}
        with open(os.path.join(self.dir, name), 'r') as f:
            for name, lable in map(lambda line: line.split(','), f.readlines()):
                png_dict[name] = lable.strip()
        return png_dict


    def __getitem__(self, index):
        """__getitem__ method.

        Args:
            index( int ): index of data to return.

        Returns:
            tuple(tensor, tensor): image and label.

        """
        name = self.png_list[index]
        label = self.png_dict[name]
        a = np.array(list(map(table, label)))

        img_path = os.path.join(self.dir, name)
        data = transforms(self.dir, name, pre=self.pre_process)
        return data, torch.tensor(a, dtype=torch.long)


    def __len__(self):
        """__len__ method.

        Returns:
            int: len. of dataset.

        """
        return len(self.png_list)


if __name__ == "__main__":    
    t = Data(train=True)
    x  = t[0]
    pass