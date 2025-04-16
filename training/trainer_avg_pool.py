# Import modules
import sys
sys.path.append("/data484_1/sislam6/practicum")

import torch
from matplotlib import pyplot as plt
from importlib import reload
from data.dataloader import DataLoader
from training.train_val_obj_avg_pool import TrainValObjAvgPool
from utilities.display import DisplayPredictions
from utilities.plot_losses import PlotLosses
import argparse
import itertools
import numpy as np
import pickle
import os

class TrainerAvgPool():
    def __init__(self, model, train_data, val_data, optimizer, scheduler, gpu_id, \
                 save_every, print_interval, max_epochs, save_model_file, save_features_file, save_model_dir, save_features_dir, display_every):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.print_interval = print_interval
        self.max_epochs = max_epochs
        self.save_model_file = save_model_file
        self.save_features_file = save_features_file
        self.save_model_dir = save_model_dir
        self.save_features_dir = save_features_dir
        self.display_every = display_every
        self.model.to(self.gpu_id)
        self.train_losses = []
        self.val_losses = []
        self.latent_features = []
        #self.predictions = []
        #self.batch_mask = []
        #self.recon_loss = []
        #self.img_list = []
        self.min_lr = 1e-05

    def _min_lr(self, lr):
        if lr < self.min_lr:
            return self.min_lr
        else:
            return lr

    def _run_epoch(self, epoch):
        # create directories to save model and features, if they don't exist
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        if not os.path.exists(self.save_features_dir):
            os.makedirs(self.save_features_dir)
        
        self.optimizer.param_groups[0]['lr'] = self._min_lr(self.optimizer.param_groups[0]['lr']) # LR cannot go below 0.00001
        print(f"Epoch [{epoch+1}/{self.max_epochs}], Learning Rate: {self.optimizer.param_groups[0]['lr']}")

        self.model.train() # set model to training mode
        train_total_loss = 0.0
        val_total_loss = 0.0
        count = 0
        # save embeddings
        self.latent_features = [] # will be replaced every epoch. just saving from last epoch
        # save reconstruction loss for dataset. Generally will be done with test data. 
        #self.recon_loss = [] # will be replaced every epoch. just saving from last epoch
        #self.img_list = [] # will be replaced every epoch. just saving from last epoch
        #self.predictions = []
        #self.batch_mask = []
        for img, mask, img_name in self.train_data:
        #for img, mask, img_name in itertools.islice(self.train_data, 5):
            
            img, mask = img.to(self.gpu_id), mask.to(self.gpu_id)   
            latent, loss, predictions, batch_mask = self.model(img, mask)
            #self.recon_loss.append(loss)
            #self.img_list.append(img_name)
            self.latent_features.append(latent.cpu().detach().numpy())
            #self.predictions.append(predictions.cpu().detach().numpy())
            #self.batch_mask.append(batch_mask.cpu().detach().numpy())
            train_total_loss += loss
            count += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if count % self.print_interval == 0:
                print(f"Batch {count}/{len(self.train_data)}, Loss: {loss:.4f}")

        #print(np.array(self.latent_features).shape)

        avg_train_loss = train_total_loss/len(self.train_data)
        print(f"Epoch [{epoch+1}/{self.max_epochs}], Training Complete. Average Training Loss: {avg_train_loss:.4f}")

        # evaluate on validation set
        self.model.eval()
        with torch.no_grad():
            for val_img, val_mask, val_img_name in self.val_data:
            #for val_img, val_mask, val_img_name in itertools.islice(self.val_data, 5):
                val_img, val_mask = val_img.to(self.gpu_id), val_mask.to(self.gpu_id)
                _, val_loss, val_predictions, val_batch_mask = self.model(val_img, val_mask)
                val_total_loss += val_loss

        avg_val_loss = val_total_loss/len(self.val_data)

        self.scheduler.step()

        print(f'Epoch [{epoch+1}/{self.max_epochs}], Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}')
        
        if epoch % self.display_every == 0 or self.display_every == self.max_epochs-1:
            DisplayPredictions(self.gpu_id,
                 predictions, 
                 batch_mask, 
                 img_size=182,
                 num_frames=224,
                 patch_size=14,
                 tubelet_size=16).disp_pred()

        # save features for GWAS
        if epoch % self.save_every == 0 or epoch == self.max_epochs-1:
            with open(f'{self.save_features_dir}/{self.save_features_file}_{epoch+1}.pkl', 'wb') as f:
                print(f"Epoch [{epoch+1}]. Saving features") #of shape: {np.array(self.latent_features).shape}")
                pickle.dump(self.latent_features, f)
        
        return avg_train_loss, avg_val_loss

    def _save_checkpoint(self, epoch):
        #checkpoint = self.model.state_dict()
        checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),   
        'training_loss': self.train_losses,
        'validation_loss': self.val_losses}
        PATH = f"{self.save_model_dir}/{self.save_model_file}_{epoch+1}.pt"
        torch.save(checkpoint, PATH)
        print(f"Epoch [{epoch+1}]. Model saved to {PATH}")

    def train(self):
        for epoch in range(self.max_epochs):
            avg_train_loss, avg_val_loss = self._run_epoch(epoch)
            if (epoch % self.save_every == 0) or (epoch == self.max_epochs-1):
                self._save_checkpoint(epoch)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
        PlotLosses(self.train_losses, self.val_losses).plot_losses()

# run code with __main__

if __name__ == "__main__":
    
    pass
    

        
