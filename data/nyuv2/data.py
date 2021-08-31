import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

from PIL import Image


class NYUv2DepthDataset(Dataset):
    def __init__(self, dataset_file, model_name, use_for):
        """

        :param dataset_file:
        :param transform:
        :param use_for: which process this Dataset is used for. [train|eval|dissect]
        """
        with open(dataset_file, 'r') as f:
            self.filenames = f.readlines()
        self.root_dir = 'dataset/nyuv2/'
        self.model_name = model_name
        self.use_for = use_for

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        image_path = os.path.join(self.root_dir, sample_path.strip().split(',')[0])
        depth_path = os.path.join(self.root_dir, sample_path.strip().split(',')[1])

        sample = {}
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        image, depth = self.transform_image_depth(image, depth)
        sample['image'], sample['depth'] = image, depth

        return sample

    def transform_image_depth(self, image, depth):
        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        if 'MFF' in self.model_name:
            if self.use_for == 'train':
                image, depth = transforms.Resize(240)(image), transforms.Resize(240, Image.NEAREST)(depth)
                # RandomHorizontalFlip
                if random.random() < 0.5:
                    image, depth = TF.hflip(image), TF.hflip(depth)
                # RandomRotate(5)
                random_rotate_angle = random.uniform(-5, 5)
                image, depth = TF.rotate(image, random_rotate_angle), TF.rotate(depth, random_rotate_angle)

                image_transform = transforms.Compose([
                    transforms.CenterCrop((228, 304)),
                    transforms.ToTensor(),
                    Lighting(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
                ])
                depth_transform = transforms.Compose([
                    transforms.CenterCrop((228, 304)),
                    transforms.Resize((114, 152), Image.NEAREST),
                    Depth2Tensor()
                ])
                image, depth = image_transform(image), depth_transform(depth)
            else:
                image_transform = transforms.Compose([
                    transforms.Resize(240),
                    transforms.CenterCrop((228, 304)),
                    transforms.ToTensor(),
                    transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
                ])
                depth_transform = transforms.Compose([
                    transforms.Resize(240, Image.NEAREST),
                    transforms.CenterCrop((228, 304)),
                    Depth2Tensor()
                ])
                image, depth = image_transform(image), depth_transform(depth)
        elif 'BTS' in self.model_name:
            if self.use_for == 'train':
                image, depth = transforms.CenterCrop((427, 565))(image), transforms.CenterCrop((427, 565))(depth)
                # RandomRotate(2.5)
                random_rotate_angle = random.uniform(-2.5, 2.5)
                image, depth = TF.rotate(image, random_rotate_angle), TF.rotate(depth, random_rotate_angle)
                # RandomCrop(416, 544)
                x = random.randint(0, image.width - 544)
                y = random.randint(0, image.height - 416)
                image, depth = TF.crop(image, y, x, 416, 544), TF.crop(depth, y, x, 416, 544)
                # RandomHorizontalFlip
                if random.random() < 0.5:
                    image, depth = TF.hflip(image), TF.hflip(depth)
                image_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
                ])
                depth_transform = transforms.Compose([
                    Depth2Tensor()
                ])
                image, depth = image_transform(image), depth_transform(depth)
            else:
                image_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
                ])
                depth_transform = transforms.Compose([
                    Depth2Tensor()
                ])
                image, depth = image_transform(image), depth_transform(depth)
        else:
            raise NotImplementedError

        return image, depth

    def __len__(self):
        return len(self.filenames)


def get_dataloader(use_for, dataset_file, model_name, batch_size, seed=None, shuffle=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    dataset = NYUv2DepthDataset(dataset_file, model_name, use_for)
    if shuffle is None:
        shuffle = True if use_for == 'train' else False
    N_CPU = 4
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=N_CPU, pin_memory=False)

    return dataloader


class Depth2Tensor(object):
    """
    Convert depth to a tensor.
    """
    def __call__(self, depth):
        depth = TF.to_tensor(depth).float()
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
        if depth.max() > 1:
            depth = depth / 1000
        else:
            depth = depth * 10
        return depth


class Lighting(object):
    def __init__(self):
        self.alphastd = 0.1
        __imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        }
        self.eigval = __imagenet_pca['eigval']
        self.eigvec = __imagenet_pca['eigvec']

    def __call__(self, image):
        if self.alphastd == 0:
            return image

        alpha = image.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(image).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        image = image.add(rgb.view(3, 1, 1).expand_as(image))

        return image
