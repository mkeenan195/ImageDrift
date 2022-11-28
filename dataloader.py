import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode


class ImagingDataset(Dataset):
    '''Thermal Imaging Dataset'''
    
    def __init__(self, image_dir, start_date, end_date, transform=None):
        '''
        Parameters
        ----------
        image_dir : str
            File path to folder with images.
        start_date : str
            Start date for images in format MMDD.
        end_date : str
            End date for images in format MMDD.
        '''
        self.image_dir = image_dir
        files = [x for x in os.listdir(image_dir) if (x[4:8]>=start_date and x[4:8]<=end_date)]
        self.image_names = list(set([x.split(".")[0] for x in files]))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image_name = self.image_names[idx]
        image_path = f'{self.image_dir}/{image_name}.jpg'

        img = read_image(image_path, mode=ImageReadMode.GRAY)
        img = img / 255

        if self.transform:
            img = self.transform(img)

        return img
        
        
        
        




