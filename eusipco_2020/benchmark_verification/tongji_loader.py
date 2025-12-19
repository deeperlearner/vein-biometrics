import os
import glob
from pathlib import Path, PurePath
import random

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import PIL.Image
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


class SingleReader(Dataset):
    def __init__(self, database_dir, mode="train", transform=None):
        self.database_dir = database_dir
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        # if self.mode == "train":
        #     # index: 0~7199
        #     # palm: 0~359
        #     palm = index // 20
        # elif self.mode == "valid":
        #     # index: 0~2399
        #     # palm: 360~479
        #     palm = index // 20 + 360
        # # img_number: 1~20
        # img_number = index % 20 + 1

        # if img_number <= 10: # 1~10
        #     number = palm * 10 + img_number
        #     im_file = os.path.join(self.database_dir, f"session1/{number:05d}.bmp")
        # else: # 11~20
        #     number = palm * 10 + img_number - 10
        #     im_file = os.path.join(self.database_dir, f"session2/{number:05d}.bmp")
        if self.mode == "train":
            # index: 0~7199
            # img_number: 1~12
            img_number = index // 600 + 1
        elif self.mode == "valid":
            # index: 0~2399
            # img_number: 13~16
            img_number = index // 600 + 13
        # palm: 0~599
        palm = index % 600
        if img_number <= 10: # 1~10
            number = palm * 10 + img_number
            im_file = os.path.join(self.database_dir, f"session1/{number:05d}.bmp")
        else: # 11~16
            number = palm * 10 + img_number - 10
            im_file = os.path.join(self.database_dir, f"session2/{number:05d}.bmp")

        img = self.im_reader(im_file)
        return img, palm, im_file

    def im_reader(self, im_file):
        im = PIL.Image.open(im_file)
        im = im.convert('RGB')
        return self.transform(im)

    def __len__(self):
        if self.mode == "train":
            return int(6000 * 2 * 0.6) # 7200
        elif self.mode == "valid":
            return int(6000 * 2 * 0.2) # 2400


class PairReader(Dataset):
    def __init__(self, database_dir, transform=None):
        self.database_dir = database_dir
        self.transform = transform

        test_data = dict()
        # palm: 0~599
        for palm in range(600):
            test_data[palm] = []
            # img_number: 17~20
            for img_number in range(17, 21):
                number = palm * 10 + img_number - 10
                test_data[palm].append(os.path.join(self.database_dir, f"session2/{number:05d}.bmp"))
        self.total_len = 150000
        self.hands = []
        for i in tqdm(range(self.total_len)):
            if i % 2 == 0:
                # genuine
                cls, hands = random.choice(list(test_data.items()))
                hand1, hand2 = random.sample(hands, 2)
                self.hands.append((hand1, hand2, 1))
            else:
                # impostor
                cls1, cls2 = random.sample(list(test_data.keys()), 2)
                hand1 = random.choice(test_data[cls1])
                hand2 = random.choice(test_data[cls2])
                self.hands.append((hand1, hand2, 0))

    def __getitem__(self, index):
        im_file1, im_file2, classname = self.hands[index]
        img1 = self.im_reader(im_file1)
        img2 = self.im_reader(im_file2)
        return img1, img2, classname, im_file1, im_file2

    def im_reader(self, im_file):
        im = PIL.Image.open(im_file)
        im = im.convert('RGB')
        return self.transform(im)

    def __len__(self):
        return self.total_len


def get_dataloader_tongji(database_dir, batch_size=32, num_workers=8):
    im_size = (128, 128)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_dataset = SingleReader(database_dir, mode="train", transform=transforms.Compose([transforms.Resize(im_size),
                                                                                           # transforms.ColorJitter(brightness=0.5),
                                                                                           transforms.RandomAffine(5, translate=(0.05, 0.05)),
                                                                                           transforms.ToTensor(),
                                                                                           normalize]))
    valid_dataset = SingleReader(database_dir, mode="valid", transform=transforms.Compose([transforms.Resize(im_size),
                                                                                           transforms.ToTensor(),
                                                                                           normalize]))
    test_dataset = PairReader(database_dir, transform=transforms.Compose([transforms.Resize(im_size),
                                                                          transforms.ToTensor(),
                                                                          normalize]))

    dataloaders = {}
    dataloaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloaders["valid"] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloaders["test"] = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloaders


if __name__ == '__main__':
    database_dir = "/media/back/home/chuck/Dataset/Tongji_ROI"
    im_size = (128, 128)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    dataset = SingleReader(database_dir, mode="train", transform=transforms.Compose([transforms.Resize(im_size),
                                                                                     transforms.ToTensor(),
                                                                                     normalize]))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    # for img, palm, im_file in dataloader:
    #     # print(im_file)
    #     break
    dataloaders = get_dataloader(database_dir)
