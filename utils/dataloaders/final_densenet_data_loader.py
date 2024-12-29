from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os
import shutil
import random


def save_data_to_folders():
    # this function is used to save the data to folders

    cifake_train_class_0 = datasets.ImageFolder(root="./CIFake/train/class_0", transform=data_transforms['train'])
    cifake_train_class_1 = datasets.ImageFolder(root="./CIFake/train/class_1", transform=data_transforms['train'])

    # set half of dataset randomly (due to computational power and time)
    cifake_class_0_indices = np.arange(len(cifake_train_class_0))
    selected_cifake_class_0_indices = np.random.choice(cifake_class_0_indices, len(cifake_train_class_0) // 2, replace=False)
    cifake_class_0_half = Subset(cifake_train_class_0, selected_cifake_class_0_indices)

    cifake_class_1_indices = np.arange(len(cifake_train_class_1))
    selected_cifake_class_1_indices = np.random.choice(cifake_class_1_indices, len(cifake_train_class_1) // 2, replace=False)
    cifake_class_1_half = Subset(cifake_train_class_1, selected_cifake_class_1_indices)

    
    # function for splitting into train and validation
    # default is 80:20 and that's what we used in all models
    def split_data(dataset):
        indices = list(range(len(dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        return train_subset, val_subset

    cifake_class_0_train, cifake_class_0_val = split_data(cifake_class_0_half)
    cifake_class_1_train, cifake_class_1_val = split_data(cifake_class_1_half)

    #put classes together (in val and train dataset)
    combined_train_data = torch.utils.data.ConcatDataset([cifake_class_0_train, cifake_class_1_train])
    combined_val_data = torch.utils.data.ConcatDataset([cifake_class_0_val, cifake_class_1_val])

    train_dir = './data/train'
    val_dir = './data/val'

    # just in case that there is no folder in branch
    for dir_path in [train_dir, val_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # saving - traindata
    for i, (image, label) in enumerate(combined_train_data):
        label_dir = os.path.join(train_dir, f"class_{label}")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        image_path = os.path.join(label_dir, f"{i}.png")
        transforms.ToPILImage()(image).save(image_path)

    # saving - valdata
    for i, (image, label) in enumerate(combined_val_data):
        label_dir = os.path.join(val_dir, f"class_{label}")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        image_path = os.path.join(label_dir, f"{i}.png")
        transforms.ToPILImage()(image).save(image_path)



#loading the data - validation and train set
def create_dataloaders(batch_size=32):
    # again this is the part which is different in every model
    #this was the densenet model with best results- we applied more data augmentation
    # we had to resize the images to 224x224 because densenet was originally trained on this size
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  #this was added crop of images
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    #loading images
    train_dataset = datasets.ImageFolder(root="./data/train", transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root="./data/val", transform=data_transforms['val'])

    #dataloaders - shuffle for training but no for vaidation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    return dataloaders


#loading test set
def create_test_loader(batch_size=32):
    
    #just resizing cause it's densenet
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    cifake_test_class_0 = datasets.ImageFolder(root="./CIFake/test/class_0", transform=data_transforms['test'])
    cifake_test_class_1 = datasets.ImageFolder(root="./CIFake/test/class_1", transform=data_transforms['test'])

    test_data = torch.utils.data.ConcatDataset([cifake_test_class_0, cifake_test_class_1])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4) #no shuffle

    return test_loader
