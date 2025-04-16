import torch
from matplotlib import pyplot as plt

class DisplayPredictions():
    def __init__(self, 
                 gpu_id,
                 predictions, 
                 batch_mask, 
                 img_size=182,
                 num_frames=224,
                 patch_size=14,
                 tubelet_size=16):
        self.gpu_id = gpu_id
        self.predictions = predictions
        self.batch_mask = batch_mask
        self.input_size = img_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

    def disp_pred(self):
        grid_size = self.input_size // self.patch_size # this is 13 (182/14)
        grid_depth = self.num_frames // self.tubelet_size # this is 14 (224/16)

        num_patches = grid_size * grid_depth * grid_size # 13 * 14 * 13
        patch_vol = self.patch_size * self.tubelet_size * self.patch_size
        
        empty_img = torch.zeros((2, num_patches, patch_vol))
        empty_img = empty_img.to(self.gpu_id)

        empty_img[:, self.batch_mask, :] = self.predictions
        recon_img = empty_img.view(2,grid_size,grid_depth,grid_size,self.patch_size,self.tubelet_size,self.patch_size).permute(0,1,4,2,5,3,6).\
        reshape(-1,self.input_size,self.num_frames,self.input_size)
        recon_img = recon_img[0]
        recon_img = recon_img.detach()

        # visualize reconstructed image
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5)) 

        recon_img = recon_img.cpu() # move back to CPU to visualize

        axes[0].imshow(recon_img[recon_img.shape[0] // 2, :, :], cmap='gray')  
        axes[0].set_title('Sagital')

        axes[1].imshow(recon_img[:, recon_img.shape[1] // 2, :], cmap='gray')
        axes[1].set_title('Frontal')

        axes[2].imshow(recon_img[:, :, recon_img.shape[2] // 2], cmap='gray')
        axes[2].set_title('Transverse')

        fig.suptitle('Reconstructed Brain Images', fontsize=16, fontweight='bold')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
