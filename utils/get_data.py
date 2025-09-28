import torch
from torch.utils.data import Subset
from torchvision import datasets
import os
import random
import numpy as np
from utils.transforms import get_transforms

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def get_datasets():
    transform = get_transforms()
    dataset_path = 'data/PlantVillage'

    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    split_file = 'split_indices.pth'

    if os.path.exists(split_file):
        split = torch.load(split_file)
        
        train_indices = split['train']
        val_indices = split['val']
        test_indices = split['test']
        
    else:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        

        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        
        train_indices = indices[: train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size: ]
        
        torch.save({
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }, split_file)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    return dataset, train_set, val_set, test_set