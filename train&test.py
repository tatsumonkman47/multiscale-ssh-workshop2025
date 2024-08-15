import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import os
import time
from utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque


############################################################   TRAINING   ####################################################################

def train_model(model, train_loader, val_loader,
                criterion, optimizer, device,
                save_path ='/home/jovyan/SSH/B_data/updated_dm/test3/model.pth', n_epochs=2000):

    """
    This function trains a deep learning model.
    
    Parameters:
    - model: The neural network model to be trained
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - criterion: Loss function
    - optimizer: Optimization algorithm
    - device: Device to run the model on (CPU or GPU)
    - save_path: Path to save the trained model
    - n_epochs: Number of training epochs
    """
    
    def print_numberofparameters(model):
        """Print the number of trainable parameters in the model"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params}")
    
    # Create a learning rate scheduler that reduces the learning rate when the validation loss stops improving    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=4, verbose=True)

    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    print_numberofparameters(model)
    
    # Initialize variables for early stopping
    k = 5  # Number of best validation losses to keep track of
    best_val_loss = deque(maxlen=k)
    patience = 50   # Number of epochs to wait before early stopping
    patience_counter = 0
    
    # Check if a saved model exists and load it if it does
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"Resuming from epoch {start_epoch} with best val losses {list(best_val_loss)}")
    else:
        start_epoch = 0

    # Main training loop
    for epoch in range(start_epoch, n_epochs):
        
        start_time = time.time()  
        model.train()  # Set model to training mode
        train_running_loss = 0.0

        # Training phase
        for batch_x, batch_y in train_loader:
            # Move batch to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item() * batch_x.size(0)
            
        # Calculate average training loss for the epoch
        epoch_loss = train_running_loss / len(train_loader.dataset)

        
        # Validation phase
        model.eval()   # Set model to evaluation mode
        val_running_loss = 0.0
    
        with torch.no_grad():   # Disable gradient calculation for validation
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_running_loss += loss.item() * batch_x.size(0)
                
        # Calculate average validation loss
        val_loss = val_running_loss / len(val_loader.dataset)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Calculate epoch duration and peak memory usage
        end_time = time.time()
        epoch_duration = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.3e}, Val Loss: {val_loss:.3e}, Epoch Time: {epoch_duration:.2f}s')
        
        # Save model for first 30 epochs regardless of performance
        if epoch < 30:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint, save_path)
            print(f'Model saved at epoch {epoch+1}')
        else:
            # After 30 epochs, use early stopping logic
            if len(best_val_loss) < k or val_loss < max(best_val_loss):
                if len(best_val_loss) == k:
                    best_val_loss.remove(max(best_val_loss))
                best_val_loss.append(val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'Patience counter: {patience_counter}/{patience}')
            
            # Save checkpoint only if it's the best model so far
            if val_loss == min(best_val_loss):
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter
                }
                torch.save(checkpoint, save_path)
                print(f'Best model so far saved to {save_path}')
            
            # Early stopping check
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

    print('Training complete')


############################################################   TESTING   ####################################################################


def evaluate_model(model, device, test_loader, ssh_test, checkpoint_path):

    """
    Evaluate a trained model on test data and generate predictions.

    Parameters:
    - model: The neural network model to be evaluated
    - device: Device to run the model on (CPU or GPU)
    - test_loader: DataLoader containing the test data
    - ssh_test: Sea Surface Height test data (xarray DataArray)
    - checkpoint_path: Path to the saved model checkpoint

    Returns:
    - bm_prediction: Balanced motion prediction (SSH_test - UBM_prediction)
    - ubm_prediction: Unbalanced motion prediction
    """
    
    # Move the model to the specified device (CPU or GPU)
    model = model.to(device)
    # Set the model to evaluation mode
    model.eval()  
    
    # Load the trained model parameters from the checkpoint
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model parameters from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    predictions = []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:  
            batch_x = batch_x.to(device)
            y_pred = model(batch_x)
            predictions.append(y_pred.cpu())
            
    # Combine all batch predictions and convert to numpy array
    prediction = (torch.cat(predictions, dim=0)).squeeze(1).numpy() 

    # Convert predictions to xarray DataArray with the same structure as ssh_test
    ubm_prediction = xr_da(prediction, ssh_test)
    
    bm_prediction = ssh_test - ubm_prediction
    
    return bm_prediction, ubm_prediction


def evaluate_model_zca(model, device, test_loader, zca_matrix, data_mean, ssh_test, checkpoint_path):
    """
    Evaluate a trained model on test data and generate predictions, with ZCA whitening.

    Parameters:
    - model: The neural network model to be evaluated
    - device: Device to run the model on (CPU or GPU)
    - test_loader: DataLoader containing the test data
    - zca_matrix: ZCA whitening matrix for inverse transformation
    - data_mean: Mean of the data used for ZCA whitening
    - ssh_test: Sea Surface Height test data (xarray DataArray)
    - checkpoint_path: Path to the saved model checkpoint

    Returns:
    - bm_prediction: Balanced motion prediction (SSH_test - UBM_prediction)
    - ubm_prediction: Unbalanced motion prediction
    """    
    model = model.to(device)
    model.eval()  

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model parameters from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    predictions = []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:  
            batch_x = batch_x.to(device)
            y_pred = model(batch_x)
            predictions.append(y_pred.cpu())

    prediction_np = (torch.cat(predictions, dim=0)).squeeze(1).numpy() 
    
    # Apply inverse ZCA transformation to the predictions
    prediction = inverse_zca_transformation(prediction_np, zca_matrix, data_mean)
    
    # Convert predictions to xarray DataArray with the same structure as ssh_test
    ubm_prediction = xr_da(prediction, ssh_test)
    
    bm_prediction = ssh_test - ubm_prediction
    
    return bm_prediction, ubm_prediction



