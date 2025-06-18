# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import torch
import utils
import scipy.io
import numpy as np
import pandas as pd

from PIL import Image
from samplers import RASampler
from timm.data import create_transform
from torch.utils.data import Subset, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            path_current = os.path.join(root, elem['file_name'])
            target_current = int(elem['file_name'].split('/')[2])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]

            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class Flowers102Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform

        labels_path = os.path.join(root, "imagelabels.mat")
        labels = scipy.io.loadmat(labels_path)["labels"][0] - 1 

        image_dir = os.path.join(root, "jpg")
        image_files = sorted(os.listdir(image_dir))
        self.image_paths = [os.path.join(image_dir, img) for img in image_files]
        self.labels = labels

        split_path = os.path.join(root, "setid.mat")
        split_data = scipy.io.loadmat(split_path)

        if train:
            split_indices = np.concatenate((split_data["trnid"][0], split_data["valid"][0])) - 1
        else:
            split_indices = split_data["tstid"][0] - 1

        self.image_paths = [self.image_paths[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


class StanfordCarsDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.train = train

        file_path = os.path.join(root, "cars_info.xlsx")
        train_path = "cars_train/cars_train/"
        test_path = "cars_test/cars_test/"
        img_folder = os.path.join(root, train_path if train else test_path)
        
        xls = pd.ExcelFile(file_path)
        df = xls.parse("train" if train else "test")

        for _, row in df.iterrows():
            img_name = row["image"].strip("'")
            img_path = os.path.join(img_folder, img_name)
            label = int(row["class"]) - 1
            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == "CIFAR100":
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
        
    elif args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
        
    elif args.dataset == "IMNET":
        if is_train:
            root = os.path.join(args.data_path, "ILSVRC2012_img_train")
        else:
            root = "/home/u17/yuxinr/datasets/ILSVRC2012_img_val"
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
        if is_train and args.train_subset < 1.0:
            labels = np.array([dataset.targets[i] for i in range(len(dataset))])
            sss = StratifiedShuffleSplit(n_splits=1, train_size=args.train_subset, random_state=42)
            indices, _ = next(sss.split(np.zeros(len(labels)), labels))
            dataset = Subset(dataset, indices)
            
    elif args.dataset == "INAT18":
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
        
    elif args.dataset == "INAT19":
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    elif args.dataset == 'FLOWER':
        dataset = Flowers102Dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 102

    elif args.dataset == 'CAR':
        dataset = StanfordCarsDataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 196
        
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def load_dataset(args, mode):
    if mode == "train":
        print(f"Loading training dataset {args.dataset}")
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            if args.repeated_aug:
                sampler_train = RASampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        return data_loader_train
    
    elif mode == "val":
        print(f"Loading validation dataset {args.dataset}")
        dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        return data_loader_val, dataset_val

