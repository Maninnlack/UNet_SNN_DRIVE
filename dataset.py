import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from utils import transforms as T


class DriveDataset(Dataset):
    def __init__(self, root, dtype='test', transforms=None):
        super().__init__()
        self.root = root
        image_path = os.path.join(self.root, dtype, 'images')
        assert os.path.exists(image_path), f"path {image_path} does not exists."
        image_list = os.listdir(image_path)
        self.image_path_list = [os.path.join(image_path, i) for i in image_list if i.endswith('.tif')]

        mask_path = os.path.join(self.root, dtype, 'mask')
        self.mask_path_list = [os.path.join(mask_path, i.split('.')[0] + '_mask.gif') for i in image_list]

        manual_path = os.path.join(self.root, dtype, '1st_manual')
        self.manual_path_list = [os.path.join(manual_path, i.split('_')[0] + '_manual1.gif') for i in image_list]

        #self.times = times    # 数据扩增倍数
        self.transforms = transforms


    def __len__(self):
        return len(self.image_path_list) # * self.times

    def __getitem__(self, index):
        
        image = Image.open(self.image_path_list[index]).convert('RGB')
        manual = Image.open(self.manual_path_list[index]).convert('L')
        roi_mask = Image.open(self.mask_path_list[index]).convert('L')
        
        manual = np.array(manual) / 255
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        # 此时的感兴趣区域中的前景像素值=1，后景=0，不感兴趣区域像素值=255

        mask = Image.fromarray(mask)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)
    
    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 256

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)



if __name__ == '__main__':
    data_root = './data'
    
    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    train_dataset = DriveDataset(data_root, 
                                 dtype='train', 
                                 transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset = DriveDataset(data_root,
                               dtype='val', 
                               transforms=get_transform(train=False, mean=mean, std=std))
