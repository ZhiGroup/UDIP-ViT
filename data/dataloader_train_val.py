# this uses training set as training data and validation set as validation data
import torch
from data.dataset import aedataset

# class to prepare dataset and dataloader, both for training and validation
class DataLoaderTrainVal():
    def __init__(self, batch_size, train_modality, train_datafile, val_datafile):
        self.batch_size = batch_size
        self.train_modality = train_modality
        self.train_datafile = train_datafile
        self.val_datafile = val_datafile

    def prepare_data(self):
        T1_train_ds = aedataset(
                datafile=self.train_datafile,
                modality=self.train_modality,
            )

        T1_val_ds = aedataset(
                datafile=self.val_datafile,
                modality=self.train_modality,
            )
        
        train_dataloader = torch.utils.data.DataLoader(
                T1_train_ds, batch_size=self.batch_size, pin_memory=True, num_workers=10, shuffle=False
            )

        val_dataloader = torch.utils.data.DataLoader(
                T1_val_ds, batch_size=self.batch_size, pin_memory=True, num_workers=10, shuffle=False
            )

        return train_dataloader, val_dataloader
