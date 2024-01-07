import torch
import pydicom
import numpy as np

from torchvision.models import efficientnet_b4
from typing import List, Tuple
from copy import deepcopy
from tqdm.auto import tqdm


def collater(
        data:List[Tuple[torch.Tensor, int, str]],
        device: torch.device,
        return_prediction_ids: bool = False
    ):
    """Custom collater for managing varying size batches. Here, we stack all
    tensors on top of each-other while maintaining an index to keep track of 
    samples originating from the same prediction id.
    
    Args:
        data (List of Tuples): Loaded data containing processed images for a
            given prediction ID, the label for the prediction ID, and the string
            corresponding to the prediction ID
    
    Returns:
        images (torch.Tensor):
        labels (torch.Tensor):
        id_indices (List of Tuples): 
        prediction_ids (List of Strings):
    """
    images, label, prediction_id = data[0]
    n_images = images.shape[0]
    labels = [label] * n_images
    prediction_ids = [prediction_id] * n_images
    id_indices = [(0, images.shape[0])]
    
    for i, (image_stack, label, prediction_id) in enumerate(data[1:], start=1):
        n_images = image_stack.shape[0]
        index_start = id_indices[i-1][1]
        index_end = index_start + n_images
        
        images = torch.cat((images, image_stack), 0)
        labels.extend([label] * n_images)
        prediction_ids.extend([prediction_id] * n_images)
        id_indices.append((index_start, index_end))
    
    # Move images and Labels to the right device
    labels = torch.Tensor(labels)
    labels = labels.reshape(-1, 1)
    labels = labels.to(device)
    
    if return_prediction_ids:
        return (images, labels, id_indices, prediction_ids)
    
    return (images, labels, id_indices)


class EarlyStopper():
    def __init__(self, patience: int = 3, min_delta: float = 0.01):
        """Determine whether to proceed to the epoch based on whether the
        validation loss has increased over a given number of epochs.
        
        Args:
            patience (int): The number of epochs to wait until the loss has
                improved
            min_delta (float): The minimum decrease in loss required to satisfy
                the improvement condition
        """
        self.patience = patience
        self.min_delta = min_delta
        self.patience_remaining = patience
        self.min_loss = np.inf
    
    def check_continue(self, loss:float):
        """Check whether we should stop training the model.
        
        Args:
            loss (float): The current loss to test for improvement
        
        Returns:
            continue (bool): Whether to continue training the model
        """        
        if (loss + self.min_delta) < self.min_loss:
            self.patience_remaining = self.patience
            self.min_loss = loss
            
        if self.patience_remaining <= 0:
            return False
        
        self.patience_remaining -= 1
        return True


def train_model(
        model:efficientnet_b4,
        criterion,
        optimizer,
        scheduler,
        early_stopper,
        dataloader_train,
        dataloader_val,
        device: torch.device,
        epochs: int = 10,
    ):
    """Train and evaluate the EfficientNet-B4 model.
    """
    model = model.to(device)
    
    best_weights = deepcopy(model.state_dict())
    best_loss = np.inf
    history = {"train": [], "validation": []}
    
    for epoch in range(epochs):
        for phase in history.keys():
            phase_loss = []
            phase_mean_loss = 0.0
            
            train = phase == "train"
            model.train(train)
            if train:
                dataloader = dataloader_train
            else:
                dataloader = dataloader_val
            
            # Make classifications
            pbar_desc = f"(Epoch {epoch + 1}/{epochs}) ({phase.capitalize()})"
            pbar_dataloader = tqdm(dataloader, desc=pbar_desc)
            for images, labels, id_indices, prediction_ids in pbar_dataloader:
                optimizer.zero_grad()
                with torch.set_grad_enabled(train):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    if train:
                        loss.backward()
                        optimizer.step()
                
                phase_loss.append(loss.item())
                phase_mean_loss = np.mean(phase_loss)
                pbar_dataloader.set_description(
                    f"(Epoch {epoch + 1}/{epochs}) ({phase.capitalize()} Loss = {phase_mean_loss:.4f})"
                )
            
            # Update our training history
            history[phase].append(phase_mean_loss)
            
            if not train:
                # Check whether to reduce learning rate
                scheduler.step(phase_mean_loss)
                
                # Check if our loss improved
                if phase_mean_loss < best_loss:
                    best_loss = phase_mean_loss
                    best_weights = deepcopy(model.state_dict())
                    
                # Check whether to early stop
                if not early_stopper.check_continue(phase_mean_loss):                    
                    model.load_state_dict(best_weights)
                    return (model, history)
    
    # Return the best weights and history
    model.load_state_dict(best_weights)
    return (model, history)