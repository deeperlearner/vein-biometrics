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


class DataReader(Dataset):
    def __init__(self, hands_path, handsinfo_path, transform=None):
        self.hands_path = Path(hands_path)
        self.transform = transform
        self.df = pd.read_csv(handsinfo_path)
        self.df = self.df[self.df['aspectOfHand'].str.contains("palmar")]
        self.df = self.df.reset_index(drop=True)

    def make_split(self):
        base_path = os.path.dirname(self.hands_path)
        train_path = os.path.join(base_path, "train.csv")
        valid_path = os.path.join(base_path, "valid.csv")
        test_path = os.path.join(base_path, "test.csv")
        N = len(self)
        samples_array = np.arange(N)
        train_idx, test_idx = train_test_split(samples_array, test_size=0.2)
        train_df = self.df.iloc[train_idx]
        test_df = self.df.iloc[test_idx]

        train_id = {}
        for index, row in train_df.iterrows():
            if row["id"] not in train_id:
                train_id[row["id"]] = len(train_id)
            train_df.at[index, "id"] = train_id[row["id"]]
        test_id = {}
        for index, row in test_df.iterrows():
            if row["id"] not in test_id:
                test_id[row["id"]] = len(test_id)
            test_df.at[index, "id"] = test_id[row["id"]]

        N = len(train_df)
        samples_array = np.arange(N)
        train_idx, valid_idx = train_test_split(samples_array, test_size=0.2)
        train_df_train = train_df.iloc[train_idx]
        train_df_valid = train_df.iloc[valid_idx]

        train_df_train.to_csv(train_path, index=False)
        train_df_valid.to_csv(valid_path, index=False)
        test_df.to_csv(test_path, index=False)

    def __getitem__(self, index):
        im_file = self.df['imageName'][index]
        img = self.im_reader(self.hands_path.joinpath(im_file))
        classname = int(self.df['id'][index])
        return img, classname, im_file

    def im_reader(self, im_file):
        im = PIL.Image.open(im_file)
        im = im.convert('RGB')
        return self.transform(im)

    def __len__(self):
        return len(self.df)


class SingleReader(Dataset):
    def __init__(self, hands_path, csv_path, transform=None):
        self.hands_path = Path(hands_path)
        self.transform = transform
        self.df = pd.read_csv(csv_path)

    def __getitem__(self, index):
        im_file = self.df['imageName'][index]
        img = self.im_reader(self.hands_path.joinpath(im_file))
        classname = int(self.df['id'][index])
        return img, classname, im_file

    def im_reader(self, im_file):
        im = PIL.Image.open(im_file)
        im = im.convert('RGB')
        return self.transform(im)

    def __len__(self):
        return len(self.df)


class PairReader(Dataset):
    def __init__(self, hands_path, csv_path, transform=None):
        self.hands_path = Path(hands_path)
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.test_data = {}
        for index, row in self.df.iterrows():
            if row["id"] not in self.test_data:
                self.test_data[row["id"]] = []
            self.test_data[row["id"]].append(row["imageName"])

        self.total_len = 150000
        self.hands = []
        for i in tqdm(range(self.total_len)):
            if i % 2 == 0:
                # genuine
                hands = []
                while(len(hands) < 2):
                    cls, hands = random.choice(list(self.test_data.items()))
                hand1, hand2 = random.sample(hands, 2)
                self.hands.append((hand1, hand2, 1))
            else:
                # impostor
                cls1, cls2 = random.sample(list(self.test_data.keys()), 2)
                hand1 = random.choice(self.test_data[cls1])
                hand2 = random.choice(self.test_data[cls2])
                self.hands.append((hand1, hand2, 0))

    def __getitem__(self, index):
        im_file1, im_file2, classname = self.hands[index]
        img1 = self.im_reader(self.hands_path.joinpath(im_file1))
        img2 = self.im_reader(self.hands_path.joinpath(im_file2))
        return img1, img2, classname, str(self.hands_path.joinpath(im_file1)), str(self.hands_path.joinpath(im_file2))

    def im_reader(self, im_file):
        im = PIL.Image.open(im_file)
        im = im.convert('RGB')
        return self.transform(im)

    def __len__(self):
        return self.total_len


def get_dataloader(database_dir, batch_size=32, num_workers=8):
    train_csv = os.path.join(database_dir, "train.csv")
    valid_csv = os.path.join(database_dir, "valid.csv")
    test_csv = os.path.join(database_dir, "test.csv")
    Hands_path = os.path.join(database_dir, "Hands_crop/")
    im_size = (128, 128)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_dataset = SingleReader(Hands_path, train_csv, transforms.Compose([transforms.Resize(im_size),
                                                                            # transforms.ColorJitter(brightness=0.5),
                                                                            transforms.ToTensor(),
                                                                            normalize]))
    valid_dataset = SingleReader(Hands_path, valid_csv, transforms.Compose([transforms.Resize(im_size),
                                                                            transforms.ToTensor(),
                                                                            normalize]))
    test_dataset = PairReader(Hands_path, test_csv, transforms.Compose([transforms.Resize(im_size),
                                                                        transforms.ToTensor(),
                                                                        normalize]))

    dataloaders = {}
    dataloaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloaders["valid"] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloaders["test"] = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloaders


if __name__ == '__main__':
    database_dir = "/media/back/home/chuck/11K_Hands_processed"
    Hands_path = os.path.join(database_dir, "Hands_crop/")
    HandInfo_path = os.path.join(database_dir, "HandInfo.txt")
    im_size = (128, 128)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    dataset = DataReader(Hands_path, HandInfo_path, transforms.Compose([transforms.Resize(im_size),
                                                                        transforms.ToTensor(),
                                                                        normalize]))
    dataset.make_split()

    data_loaders = get_dataloader(database_dir)
