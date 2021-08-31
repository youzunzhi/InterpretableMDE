import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image


class KittiDepthDataset(Dataset):
    def __init__(self, dataset_file, model_name, use_for):
        with open(dataset_file, 'r') as f:
            self.filenames = f.readlines()
        self.root_dir = 'dataset/kitti/kitti/'
        self.model_name = model_name
        self.use_for = use_for

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        image_path = os.path.join(self.root_dir, sample_path.strip().split()[0])
        depth_path = os.path.join(self.root_dir, sample_path.strip().split()[1])

        sample = {}
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        focal = float(sample_path.split()[2])
        image, depth = self.transform_image_depth(image, depth)
        sample['image'], sample['depth'] = image, depth
        sample['focal'] = focal

        return sample

    def transform_image_depth(self, image, depth):
        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        if 'BTS' in self.model_name:
            if self.use_for == 'train':
                image, depth = KittiBenchmarkCrop()(image, depth)
                # RandomRotate(2.5)
                random_rotate_angle = random.uniform(-2.5, 2.5)
                image, depth = TF.rotate(image, random_rotate_angle), TF.rotate(depth, random_rotate_angle)
                # RandomCrop(352, 704)
                x = random.randint(0, image.width - 704)
                y = random.randint(0, image.height - 352)
                image, depth = TF.crop(image, y, x, 352, 704), TF.crop(depth, y, x, 352, 704)
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
                image, depth = KittiBenchmarkCrop()(image, depth)
                image_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
                ])
                depth_transform = transforms.Compose([
                    Depth2Tensor()
                ])
                image, depth = image_transform(image), depth_transform(depth)
        return image, depth

    def __len__(self):
        return len(self.filenames)


def get_dataloader(use_for, dataset_file, model_name, batch_size, seed=None, shuffle=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    dataset = KittiDepthDataset(dataset_file, model_name, use_for)
    if shuffle is None:
        shuffle = True if use_for == 'train' else False
    N_CPU = 4
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=N_CPU, pin_memory=False)

    return dataloader


class KittiBenchmarkCrop(object):
    def __init__(self):
        self.height = 352
        self.width = 1216

    def __call__(self, image, depth):
        assert image.height == depth.height
        assert image.width == depth.width
        top_margin = int(image.height - 352)
        left_margin = int((image.width - 1216) / 2)
        image = image.crop((left_margin, top_margin, left_margin+self.width, top_margin+self.height))
        depth = depth.crop((left_margin, top_margin, left_margin+self.width, top_margin+self.height))

        return image, depth


class Depth2Tensor(object):
    def __call__(self, depth):
        depth = TF.to_tensor(depth).float()
        if depth.max() > 100:  # using groundtruth
            depth = depth / 256.
        return depth