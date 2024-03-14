from torchvision import datasets, transforms
import os
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

trans = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet mean and std
])

mask_trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

# datapath = '/home/dao2/defect_detection/VisA/visa_pytorch'
# category = 'capsules'
#

def get_dataset(datapath, category, batch_size=16, data_name='mvtec'):
    assert data_name in ['mvtec', 'visa']
    assert category in os.listdir(datapath)
    train_path = os.path.join(datapath, category, 'train')
    test_path = os.path.join(datapath, category, 'test')
    assert 'good' in os.listdir(test_path) and 'good' in os.listdir(train_path)
    train_dataset = datasets.ImageFolder(train_path, transform=trans)

    # change multi-class to binary class
    classes = os.listdir(test_path)
    classes.sort()
    test_dataset = datasets.ImageFolder(test_path, transform=trans)
    test_dataset.samples = [(d,0) if x == classes.index('good') else (d,1) for d,x in test_dataset.samples]


    # get dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_mask_dataset(datapath, category, batch_size=16, data_name='mvtec'):
    assert data_name in ['mvtec', 'visa']
    assert category in os.listdir(datapath)
    mask_path = os.path.join(datapath, category, 'ground_truth')
    mask_dataset = datasets.ImageFolder(mask_path, transform=mask_trans)

    test_path = os.path.join(datapath, category, 'test')
    assert 'good' in os.listdir(test_path)
    test_dataset = datasets.ImageFolder(test_path, transform=trans)
    idx = [i for i in range(len(test_dataset)) if test_dataset.imgs[i][1] != test_dataset.class_to_idx['good']]
    test_dataset = torch.utils.data.Subset(test_dataset, idx)
    mask_loader = DataLoader(mask_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return mask_loader, test_loader


def get_validation_dataset(datapath, category, batch_size=16, data_name='mvtec'):
    assert data_name in ['mvtec', 'visa']
    assert category in os.listdir(datapath)
    val_path = os.path.join(datapath, category, 'val')
    image_path = os.path.join(val_path, 'images')
    mask_path = os.path.join(val_path, 'masks')
    image_dataset = datasets.ImageFolder(image_path, transform=trans)
    mask_dataset = datasets.ImageFolder(mask_path, transform=mask_trans)
    image_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
    mask_loader = DataLoader(mask_dataset, batch_size=batch_size, shuffle=False)
    return image_loader, mask_loader



