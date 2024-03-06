
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datapath = '/home/dao2/defect_detection/VisA/visa_pytorch'
cagegory = 'capsules'


def get_dataset(datapath,category,data_name='mvtec'):
    assert data_name in ['mvtec','visa']
    assert category in os.listdir(datapath)
    train_path = os.path.join(datapath, category, 'train')
    test_path = os.path.join(datapath, category, 'test')
    assert 'good' in os.listdir(test_path) and 'good' in os.listdir(train_path)
    classes = os.listdir(test_path)
    classes.sort()
    train_dataset = datasets.ImageFolder(train_path, transform=trans)
    test_dataset = datasets.ImageFolder(test_path, transform=trans)

    test_dataset.targets = [0 if x == classes.index('good') else 1 for x in test_dataset.targets]
    return train_dataset, test_dataset


def get_dataloader(train_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



