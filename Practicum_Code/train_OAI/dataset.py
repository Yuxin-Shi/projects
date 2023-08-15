"""
Dataset tools


(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import os
import csv


def get_pair(I):
    """
    Generates pair of images 128x128 from the knee joint.
    ps shows how big area should be mapped into that region.
    """
    s = I.size[0]
    # pad = int(np.floor(s/3))
    pad = 60
    ps = 128

    l = I.crop([0, pad, ps, pad+ps])
    m = I.crop([s-ps, pad, s, pad+ps])
    m = m.transpose(Image.FLIP_LEFT_RIGHT)
    
    return l, m


class KneeGradingDataset(data.Dataset):
    """
    Dataset class.
    """
    def __init__(self, dataset, split, transform, augment, stage='train', kl_matchbook=""):
        self.dataset = dataset
        self.names = split
        self.transform = transform
        self.augment = augment
        self.stage=stage
        with open(kl_matchbook, mode='r') as infile:
            reader = csv.reader(infile)
            matchdict = {rows[0]: rows[1] for rows in reader}
        self.matchbook = matchdict

    def __getitem__(self, index):
        fname = os.path.join(self.dataset, self.stage, self.names[index]).replace("\\","/")
        # print(self.names[index])
        # target = int(fname.split('/')[-1].split('_')[1])
        # target = int(fname.split('/')[-1].split('G')[1].split(' ')[0])
        # if self.stage == 'train':
        #     fname = os.path.join(self.dataset, self.stage, str(target), self.names[index])
        target = int(self.matchbook[self.names[index]])
        
        img = Image.open(fname).convert("L")
        # print(img.size)
        # We will use 8bit 
        tmp = np.array(img, dtype=float)
        # This line will cause arrays to be all 0, we will not convert to 8bit
        # img = Image.fromarray(np.uint8(255*(tmp/65535.)))
        img = Image.fromarray(tmp)

        img = self.augment(img)
        # img = img.resize((300, 300))

        l, m = get_pair(img)

        # get_pair first then augment
        # l, m = get_pair(img)
        #
        # l = self.augment(l)
        # m = self.augment(m)
        #
        l = self.transform(l)
        m = self.transform(m)

        return l, m, target, fname

    def __len__(self):
        return len(self.names)


class LimitedRandomSampler(data.sampler.Sampler):
    """
    Allows to use limited number of batches in the training
    """
    def __init__(self, data_source, nb, bs):
        self.data_source = data_source
        self.n_batches = nb
        self.bs = bs

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long()[:self.n_batches*self.bs])

    def __len__(self):
        return self.n_batches*self.bs
