import os
import PIL
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from utils import oneHotLabels

class HelenDataset(Dataset):
    def __init__(self, label_dir, img_dir):
        self.img_list=os.listdir(img_dir)
        self.label_list=os.listdir(label_dir)
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.label_list.sort()
        self.img_list.sort()
        #print(self.label_list)
        #print(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = Image.open(img_path)
        image = image.resize((256,256), resample=PIL.Image.BICUBIC)

        image = np.array(image)/255
        image = np.transpose(image,(2,0,1))
        image = torch.tensor(image, dtype=torch.float32, requires_grad=False)

        label_path = os.path.join(self.label_dir, self.label_list[idx])
        label = Image.open(label_path)
        label = label.resize((256,256), resample=PIL.Image.NEAREST)

        label = np.array(label)
        label = oneHotLabels(label)

        label = torch.tensor(label, dtype=torch.float32, requires_grad=False)

        sample = [image, label]
        return sample