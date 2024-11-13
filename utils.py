import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xrft
from scipy import linalg


def calculate_zca_params(data, epsilon=1e-5):
    """
    Calculate the ZCA (Zero-phase Component Analysis) whitening parameters.
    
    Args:
    - data (np.array): Input data of shape (n_samples, height, width)
    - epsilon (float): Small constant added to eigenvalues for numerical stability
    
    Returns:
    - zca_matrix (np.array): The ZCA whitening matrix
    - mean (np.array): The mean of the input data
    """
    data_flat = data.reshape((data.shape[0], -1))
    
    mask = ~np.isnan(data_flat)
    
    mean = np.nanmean(data_flat, axis=0)
    
    data_centered = data_flat - mean
    data_centered[~mask] = 0  
    
    cov_matrix = np.dot(data_centered.T, data_centered) / np.sum(mask, axis=0).clip(min=1)
    
    U, S, D = linalg.svd(cov_matrix)

    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    
    return zca_matrix, mean
    


def apply_zca_whitening(data, zca_matrix, mean):
    """
    Apply ZCA whitening transformation to the input data.
    
    Args:
    - data (np.array): Input data of shape (n_samples, height, width)
    - zca_matrix (np.array): The ZCA whitening matrix
    - mean (np.array): The mean used for centering the data
    
    Returns:
    - np.array: The whitened data with the same shape as the input
    """
    original_shape = data.shape
    data_flat = data.reshape((data.shape[0], -1))
    
    mask = ~np.isnan(data_flat)
    
    data_centered = data_flat - mean
    data_centered[~mask] = 0  

    data_whitened = np.dot(data_centered, zca_matrix)
    
    data_whitened[~mask] = np.nan
    
    return data_whitened.reshape(original_shape)



def inverse_zca_transformation(data_zca, zca_matrix, data_mean):
    """
    Apply inverse ZCA transformation to reconstruct the original data.
    
    Args:
    - data_zca (np.array): Whitened data of shape (n_samples, height, width)
    - zca_matrix (np.array): The ZCA whitening matrix
    - data_mean (np.array): The mean used for centering the original data
    
    Returns:
    - np.array: The reconstructed data with the same shape as the input
    """
    num_images, img_rows, img_cols = data_zca.shape
    
    data_zca_flat = data_zca.reshape(num_images, img_rows * img_cols)
    
    inverse_zca_matrix = np.linalg.inv(zca_matrix.T)
    
    data_reconstructed_flat = np.dot(data_zca_flat, inverse_zca_matrix)
    
    data_reconstructed = data_reconstructed_flat + data_mean
    
    data_reconstructed = data_reconstructed.reshape(num_images, img_rows, img_cols)
    
    return data_reconstructed


def isotropic_spectra(data):
    iso_psd = xrft.isotropic_power_spectrum(data, dim=['i', 'j'], detrend='constant', window=True)
    return iso_psd

def xr_da(data_np, data_like):
    coords = {dim: data_like.coords[dim].values for dim in data_like.dims}
    data_xr = xr.DataArray(data_np, dims=data_like.dims, coords=coords)
    return data_xr


def plot_field(data_xr, title, ax, cmap='jet', vmin=None, vmax=None, add_colorbar=False):

    data = data_xr.values
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, extend='both', orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.text(0.5, 1.05, '(m)', transform=cbar.ax.transAxes, ha='center', va='bottom')


def reconstruct_from_patches(patches, original_shape=(5, 2160, 2160), patch_size=108):
    time_steps, height, width = original_shape
    samples_per_timestep = patches.sizes['sample'] // time_steps
    
    reconstructed = np.zeros(original_shape)
    
    for t in range(time_steps):
        start_idx = t * samples_per_timestep
        end_idx = (t + 1) * samples_per_timestep
        time_patches = patches[start_idx:end_idx]
        
        patch_idx = 0
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                reconstructed[t, i:i+patch_size, j:j+patch_size] = time_patches[patch_idx].values
                patch_idx += 1
    
    return xr.DataArray(reconstructed, dims=['time', 'y', 'x'])