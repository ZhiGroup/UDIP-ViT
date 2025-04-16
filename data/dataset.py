import pandas as pd
import nibabel as nib
import torch
import numpy as np


class aedataset(torch.utils.data.Dataset):
    def __init__(self, datafile, modality):
        """
        Args:
            datafile (type: csv or list): the datafile mentioning the location of images or a list of file locations.
            modality (type: string): column containing location of modality of interest in the datafile.
            transforms (type: pytorch specific transforms): to add channel to the image and convert to tensor.
        Returns:
            img [torch tensor]: img file normalized 
            mask [torch tensor]: mask excluding background
            img_name [string]: name of the image
        """
        self.datafile = pd.read_csv(datafile)
        self.unbiased_brain = self.datafile[modality]

    def __len__(self):
        return len(self.unbiased_brain)

    def __getitem__(self, idxx=int):
        img_name = self.unbiased_brain[idxx]
        img = nib.load(img_name)
        img = img.get_fdata()
        img = torch.from_numpy(img)
        img = torch.nn.functional.pad(img, (0,0,3,3,0,0)) # padding image from 182x218x182 to 182x224x182
        # padding needs to be done before normalization
        mask = img != 0
        img = (img - img[img != 0].mean()) / img[img != 0].std()
        img = img.type(torch.float)
        mask = mask.int()
        return img, mask, img_name
