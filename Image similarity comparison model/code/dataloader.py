import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import random


class Train_Dataset(Dataset):
    def __init__(self, json_path, left_folder, right_folder, transform=None):
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.transform = transform

        # Load the data from the JSON file
        with open(json_path, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load the left image
        left_img_path = os.path.join(self.left_folder, item['left_img'])
        left_img = Image.open(left_img_path)
        if self.transform:
            left_img = self.transform(left_img)

        # Load the right images
        right_imgs_list = []
        for right_img_name in item['right_img']:
            right_img_path = os.path.join(self.right_folder, right_img_name)
            right_img = Image.open(right_img_path)
            if self.transform:
                right_img = self.transform(right_img)
            right_imgs_list.append(right_img)

        # Stack the right images to form a single tensor
        # random.shuffle(right_imgs_list)
        right_imgs_tensor = torch.stack(right_imgs_list)

        return left_img, right_imgs_tensor

'''class Train_Dataset(Dataset):

    def __init__(self, json_path, left_folder, right_folder, transform=None):
        with open(json_path, 'r') as f:
            self.data_info = json.load(f)
        
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        # Extract data from JSON
        left_img_name = self.data_info[index]['left']
        right_img_names = self.data_info[index]['rights']
        match_index = self.data_info[index]['match_index']

        # Load left image
        left_img_path = os.path.join(self.left_folder, left_img_name)
        left_img = Image.open(left_img_path)

        # Load right images
        right_imgs = []
        for right_img_name in right_img_names:
            right_img_path = os.path.join(self.right_folder, right_img_name)
            right_img = Image.open(right_img_path)
            right_imgs.append(right_img)
        
        random.shuffle(right_imgs)
        # Apply transforms if given
        if self.transform:
            left_img = self.transform(left_img)
            right_imgs = [self.transform(img) for img in right_imgs]

        right_imgs_tensor = torch.stack(right_imgs)

        return left_img, right_imgs_tensor, match_index'''

    
class Test_Dataset(Dataset):
    def __init__(self, json_path, left_folder, right_folder, transform=None):
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.transform = transform

        # Load the data from the JSON file
        with open(json_path, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load the left image
        left_img_path = os.path.join(self.left_folder, item['left_img'])
        left_img = Image.open(left_img_path)
        if self.transform:
            left_img = self.transform(left_img)
        
        left_name = item['name']

        # Load the right images
        right_imgs_list = []
        for right_img_name in item['right_img']:
            right_img_path = os.path.join(self.right_folder, right_img_name)
            right_img = Image.open(right_img_path)
            if self.transform:
                right_img = self.transform(right_img)
            right_imgs_list.append(right_img)

        # Stack the right images to form a single tensor
        right_imgs_tensor = torch.stack(right_imgs_list)
        return left_name, left_img, right_imgs_tensor