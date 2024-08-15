import numpy as np
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
    # Reshape the data to 2D (n_samples, n_features)
    data_flat = data.reshape((data.shape[0], -1))
    
    # Create a mask for non-NaN values
    mask = ~np.isnan(data_flat)
    
    # Calculate the mean of the data, ignoring NaN values
    mean = np.nanmean(data_flat, axis=0)
    
    # Center the data by subtracting the mean, ignoring NaN values
    data_centered = data_flat - mean
    data_centered[~mask] = 0  # Set NaN values to 0 after centering
    
    # Compute covariance matrix, accounting for NaN values
    cov_matrix = np.dot(data_centered.T, data_centered) / np.sum(mask, axis=0).clip(min=1)
    
    # Compute Singular Value Decomposition (SVD)
    U, S, * = linalg.svd(cov_matrix)
    
    # Compute ZCA whitening matrix
    # The whitening matrix transforms the data to have identity covariance
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
    
    # Create a mask for non-NaN values
    mask = ~np.isnan(data_flat)
    
    # Center the data using the provided mean
    data_centered = data_flat - mean
    data_centered[~mask] = 0  # Set NaN values to 0 after centering
    
    # Apply the whitening transformation
    data_whitened = np.dot(data_centered, zca_matrix)
    
    # Restore NaN values where they were originally
    data_whitened[~mask] = np.nan
    
    # Reshape back to original shape
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
    
    # Reshape the whitened data to 2D
    data_zca_flat = data_zca.reshape(num_images, img_rows * img_cols)
    
    # Compute the inverse of the ZCA matrix
    inverse_zca_matrix = np.linalg.inv(zca_matrix.T)
    
    # Apply the inverse transformation
    data_reconstructed_flat = np.dot(data_zca_flat, inverse_zca_matrix)
    
    # Add back the mean to complete the reconstruction
    data_reconstructed = data_reconstructed_flat + data_mean
    
    # Reshape back to original 3D shape
    data_reconstructed = data_reconstructed.reshape(num_images, img_rows, img_cols)
    
    return data_reconstructed

