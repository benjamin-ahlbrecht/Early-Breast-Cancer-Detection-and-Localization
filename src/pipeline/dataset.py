import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision import transforms
from processing import pipeline
from typing import Tuple

from processing import (
    read_encoded_stream,
    read_parameters,
    process_torch
)

from nvidia.dali.plugin.pytorch import feed_ndarray


class MammographyDataset(Dataset):
    def __init__(
            self,
            metadata: pd.DataFrame,
            height: int,
            width: int,
            device: torch.device,
            augment: bool = True,
        ):

        """Dataset for loading mammography images 
        
        Args:
            metadata (pd.DataFrame): The metadata file read in as a dataframe with
                the desired observations selected
            height (int): The desired height of the input images.
            width (int): The desired width of the input images. If an image is
                less than this width, it will not be resized.
            device (torch.device): The device to perform computations on
            augment (bool): Whether to perform data augmentations on the images
        """
        self.device = device
        self.metadata = metadata
        self.augment = augment
        
        self.height = height
        self.width = width
        
        self.prediction_ids = list(metadata.keys())
        self.labels = [value["cancer"] for value in metadata.values()]
        

        self.transforms = nn.Sequential(transforms.Resize((height, width)))
        if augment:
            self.transforms.append(transforms.RandomVerticalFlip(p=0.5))
            self.transforms.append(transforms.RandomHorizontalFlip(p=0.5))
            self.transforms.append(
                transforms.RandomAffine(
                    degrees=(-30, 30),
                    translate=(0.1, 0.1),
                    scale=(0.75, 1),
                    shear=(-15, 15)
                )
            )
            
        self.pipeline = pipeline(
            batch_size=1,
            num_threads=2,
            device_id=0,
            prefetch_queue_depth=1
        )
        
        self.pipeline.build()
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int, str]:
        """Retrieve a set of images for a given index corresponding to a
        prediction ID.
        
        Args:
            index (int): The [row] index of the metadata to query
        
        Returns:
            images (torch.Tensor): All processed images corresponding to the
                prediction ID
            label (int): Whether the prediction ID is labeled as cancerous (1) or
                not cancerous (0)
            prediction_id (str): The prediction ID corresponding to the index
                queried
        """
        prediction_id = self.prediction_ids[index]
        item = self.metadata[prediction_id]
        fnames = item["fname"]
        label = item["cancer"]
        
        images = []
        for fname in fnames:
            try:
                image = self.process_image_gpu(fname)
            except Exception as error:
                image = self.process_image_cpu(fname)
            
            image = self.transforms(image)
            images.append(image)
        
        images = torch.stack(images)
        return (images, label, prediction_id)
    
    def process_image_gpu(self, fname:str) -> torch.Tensor:
        """Process a DICOM file on the GPU
        """
        buffer = read_encoded_stream(fname)
        invert, lower, upper = read_parameters(fname)

        # Feed in external source parameters
        invert = np.array(invert).astype(np.int32)
        lower = np.array(lower).astype(np.float32)
        upper = np.array(upper).astype(np.float32)

        self.pipeline.feed_input("jpegs", [buffer])
        self.pipeline.feed_input("invert", [invert])
        self.pipeline.feed_input("lower", [lower])
        self.pipeline.feed_input("upper", [upper])

        # Extract a processed image
        output = self.pipeline.run()
        image = output[0][0]

        # Convert image to PyTorch Tensor
        image_torch = torch.empty(
            image.shape(),
            dtype=torch.float,
            device=self.device
        )
        feed_ndarray(
            image,
            image_torch,
            cuda_stream=torch.cuda.current_stream(device=0)
        )
        
        # Preprocess the tensor
        image_torch = process_torch(image_torch, self.device)
        return image_torch
    
    def process_image_cpu(self, fname:str) -> torch.Tensor:
        """Process a DICOM file on the CPU
        """
        dcm = pydicom.dcmread(fname)
        image = dcm.pixel_array

        invert, lower, upper = read_parameters(fname)

        image = image.astype(np.float32)
        image = np.clip(image, lower, upper)

        min_value = image.min()
        max_value = image.max()

        # TODO: See if if/else is faster/works
        image = (image - min_value) / (max_value - min_value)
        image = image * (1 - invert) + invert * (1 - image)

        image = torch.from_numpy(image)
        image = process_torch(image)
        image = image.to(self.device)
        return image

    def get_weighted_sampler(self, n_observations:int) -> WeightedRandomSampler:
        """Retrieve a weighted random sampler class for extracting images from
        unbalanced class sizes.
        
        Args:
            n_observations (int): The total number of observations to draw from
                the sampler. The number of batches in an epoch is the number of
                observations divided by the batch size.
        
        Returns:
            sampler (WeightedRandomSampler): The weighted random sampler
        """
        labels_unique, label_counts = np.unique(self.labels, return_counts=True)
        
        class_weights = [len(self) / count for count in label_counts]
        label_weights = [class_weights[label] for label in self.labels]
        
        sampler = WeightedRandomSampler(label_weights, n_observations)
        return sampler