# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:44:19 2025

@author: marjan
"""
import os
import torch
import numpy as np
import nibabel as nib
from PIL import Image
from vqvae import VQVAE
from torch import nn, optim
from scipy.ndimage import zoom
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt


class IXI_Dataset(Dataset):
    def __init__(self, nifti_dir, transform=None):
        self.nifti_dir = nifti_dir
        self.transform = transform
        
        # Get all .png file paths in the directory
        _ = os.listdir(nifti_dir)
        self.file_paths = [os.path.join(nifti_dir, f) for f in _ if f.endswith('.png')]
    
    def __len__(self):
        # Return the number of samples (NIfTI files) in the dataset
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        # grayscale (i.e. 1 channel)
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        
        return img

if __name__ == "__main__":
    
    # Initialize model.
    
    device = torch.device("cuda:0")
    use_ema = True
    
    model_args = {
        "in_channels": 1,
        "num_hiddens": 128, # in fc
        "num_downsampling_layers": 3, # can be max pooling or stride 2
        "num_residual_layers": 3,# includes conv+skip connection
        "num_residual_hiddens": 32,
        "embedding_dim": 2,
        "num_embeddings": 512,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    
    model = VQVAE(**model_args).to(device)
    
    # Initialize dataset.
    transform = transforms.Compose([
       transforms.ToTensor(),
    ])
    
    batch_size = 54
    # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation Learning".
    beta = 0.25
    lr = 3e-4
    
    
    # Update the path to your T1 test images directory
    test_dataset = IXI_Dataset(r"C:/Users/marjan/T1_dataset_2d_slices", transform=transform)

    
    # Use the DataLoader for test images
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle during testing
        num_workers=8,
    )
    mean = torch.zeros(1)
    sq_mean = torch.zeros(1)
    
   
    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()
    
    # saved model in this dir
    best_path = r'C:/Users/marjan/vqvae_morer_compressed.pth'
    
    
    def save_img_tensors_as_grid(img_tensors, nrows, filename):
        imgs = img_tensors.detach().cpu().numpy()  # Convert tensors to numpy
        imgs = np.clip(imgs, -0.5, 0.5)  # Clamp values to [-0.5, 0.5]
        imgs = (imgs + 0.5)  # Normalize to [0, 1] for display

        batch_size = imgs.shape[0]
        ncols = batch_size // nrows  # Calculate columns based on number of rows

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))  # Create figure grid

        for idx, ax in enumerate(axes.flat):
            ax.imshow(imgs[idx, 0], cmap="gray")  # Plot each image (assuming grayscale)
            ax.axis("off")  # Hide axes

        plt.tight_layout()
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")  # Save the grid
        plt.close()  # Close figure to free memory
    
                
        
    
    model.load_state_dict(torch.load(best_path, weights_only=False))
    # Generate and save reconstructions.
    model.eval()
    
    # @TODO test_loader with IndivsRobotic data
    with torch.no_grad():
        for imgs in test_loader: # imgs: (batch_size, channels, height, width)
            break # This loop fetches one batch of images from the test_loader and immediately breaks, so only the first batch is processed
    
        
        n_rows = 8
        save_img_tensors_as_grid(imgs, n_rows, "true_AD")
        inference = model(imgs.to(device))["x_recon"] # from dictionary - vqvae model output
        save_img_tensors_as_grid(inference, n_rows, "recon_AD")
          


# For AD subject
class IXI_Dataset(Dataset):
    def __init__(self, nifti_dir, transform=None):
        self.nifti_dir = nifti_dir
        self.transform = transform
        
        # Get all .png file paths in the directory
        _ = os.listdir(nifti_dir)
        self.file_paths = [os.path.join(nifti_dir, f) for f in _ if f.endswith('.png')]
    
    def __len__(self):
        # Return the number of samples (NIfTI files) in the dataset
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        # grayscale (i.e. 1 channel)
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        
        return img

if __name__ == "__main__":
    
    # Initialize model.
    
    device = torch.device("cuda:0")
    use_ema = True
    
    model_args = {
        "in_channels": 1,
        "num_hiddens": 128, # in fc
        "num_downsampling_layers": 3, # can be max pooling or stride 2
        "num_residual_layers": 3,# includes conv+skip connection
        "num_residual_hiddens": 32,
        "embedding_dim": 2,
        "num_embeddings": 512,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    
    model = VQVAE(**model_args).to(device)
    
    # Initialize dataset.
    transform = transforms.Compose([
       transforms.ToTensor(),
    ])
    
    batch_size = 54
    # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation Learning".
    beta = 0.25
    lr = 3e-4
    
    
    # Update the path to your T1 test images directory
    test_dataset = IXI_Dataset(r"C:/Users/marjan/AD_T1_dataset_2d_slices", transform=transform)
    
    # Use the DataLoader for test images
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle during testing
        num_workers=8,
    )
    mean = torch.zeros(1)
    sq_mean = torch.zeros(1)
    
   
    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()
    
    # saved model in this dir
    best_path = r'C:/Users/marjan/vqvae_morer_compressed.pth'
    
    
    def save_img_tensors_as_grid(img_tensors, nrows, filename):
        imgs = img_tensors.detach().cpu().numpy()  # Convert tensors to numpy
        imgs = np.clip(imgs, -0.5, 0.5)  # Clamp values to [-0.5, 0.5]
        imgs = (imgs + 0.5)  # Normalize to [0, 1] for display

        batch_size = imgs.shape[0]
        ncols = batch_size // nrows  # Calculate columns based on number of rows

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))  # Create figure grid

        for idx, ax in enumerate(axes.flat):
            ax.imshow(imgs[idx, 0], cmap="gray")  # Plot each image (assuming grayscale)
            ax.axis("off")  # Hide axes

        plt.tight_layout()
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")  # Save the grid
        plt.close()  # Close figure to free memory
    
                
        
    
    model.load_state_dict(torch.load(best_path, weights_only=False))
    # Generate and save reconstructions.
    model.eval()
    
    # @TODO test_loader with IndivsRobotic data
    with torch.no_grad():
        for imgs in test_loader: # imgs: (batch_size, channels, height, width)
            break # This loop fetches one batch of images from the test_loader and immediately breaks, so only the first batch is processed
    
        
        n_rows = 8
        save_img_tensors_as_grid(imgs, n_rows, "true_AD")
        inference = model(imgs.to(device))["x_recon"] # from dictionary - vqvae model output
        save_img_tensors_as_grid(inference, n_rows, "recon_AD")
        

