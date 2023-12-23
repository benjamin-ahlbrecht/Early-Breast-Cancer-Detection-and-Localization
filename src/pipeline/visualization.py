import numpy as np
import matplotlib.pyplot as plt
import torch


class ClassActivationMapper():
    def __init__(self, model, aggregator):
        """Create class activation maps (CAMs) from the EfficientNet-B4 model.
        
        Args:
            model (torch.models.efficientnet_b4): The EfficientNet-B4 model
                edited and trained for your classification task
        """
        self.model = model.eval()
        self.aggregator = aggregator
        self.model_features = model.features
        self.model_pool = model.avgpool
        self.model_classifier = model.classifier
        self.activation = torch.nn.Sigmoid()
        
        self.weights, self.bias = self.model_classifier.parameters()
    
    def class_activation_tensor(self, images:torch.Tensor) -> torch.Tensor:
        """Retrieve the raw class activation tensor showing how each spatial
        unit in the last convolutional feature map contributes exactly to the
        model's classification decision.
        
        Args:
            images (torch.Tensor): A batch of input images with dimensionality
                (B, 3, H, W). If there is only a single image, then the batch
                size is one; i.e., (1, 3, H, W)
        
        Returns:
            influence (torch.Tensor): A batch of output tensors showing the
                relative contribution each spatial unit in the feature map has
                in the model's classification decision.
        """
        with torch.no_grad():
            feature_maps = self.model_features(images.to(DEVICE))
            
            influence = torch.zeros_like(feature_maps)
            for i, feature_map in enumerate(feature_maps):
                influence[i] = (feature_map.T * self.weights).T
            
            influence = torch.sum(influence, axis=1)
            return influence
    
    def class_activation_map(self, images:torch.Tensor) -> np.ndarray:
        """Retrieve the class activation map for a batch of images being
        classified by the model. Shows the relative contribution each pixel has
        in the model's classification decision.
        
        Args:
            images (torch.Tensor): A batch of input images with dimensionality
                (B, 3, H, W). If there is only a single image, then the batch
                size is one; i.e., (1, 3, H, W)
        
        Returns:
            activation_maps (np.ndarray): The class activation map for each
                image in the batch
        """
        batch_size, channels, height, width = images.shape
        resizer = transforms.Resize((height, width))
        
        influence = self.class_activation_tensor(images)
        
        activation_map = resizer.forward(influence)
        return activation_map.cpu().numpy()
    
    def patient_single_map(self, images:torch.Tensor):
        """
        """
        n_images = images.shape[0]
        cams = self.class_activation_map(images)
        
        # Align the CAMs as a single tensor
        cam = np.concatenate([cam for cam in cams], axis=1)
        images = np.concatenate([image[0].cpu().numpy() for image in images], axis=1)
        
        # To avoid a weird overlay, threshold the CAM
        cam_mean = np.mean(cam)
        cam_std = np.std(cam)
        cam_mask = cam < cam_mean + cam_std
        cam[cam_mask] = 0

        # Also, we'll make the opacity a function of CAM values
        cam_min = np.min(cam)
        cam_max = np.max(cam)
        alphas = (cam - cam_min) / (2 * (cam_max - cam_min))
        alphas = np.nan_to_num(alphas)
        
        fig, ax = plt.subplots(figsize=(10*n_images, 10), tight_layout=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(images, cmap="gray")
        ax.imshow(cam, cmap="inferno", alpha=alphas)
        
        return fig, ax
    
    def patient_maps(
            self, images:torch.Tensor, label:int, prediction_id:str):
        """
        """
        n_images = images.shape[0]
        cams = self.class_activation_map(images)
        
        # Make individual predictions
        with torch.no_grad():
            predictions = self.model(images)
            
        prediction_aggregated = int(round(self.aggregator(predictions.cpu().numpy())))
                
        # Convert our label to a string
        if label == 0:
            label = "Benign"

        if label == 1:
            label = "Malignant"
        
        if prediction_aggregated == 0:
            prediction_aggregated = "Benign"
        if prediction_aggregated == 1:
            prediction_aggregated = "Malignant"
        
        
        fig, ax = plt.subplots(ncols=n_images, tight_layout=True)
        fig.suptitle(f"Truth: {label}\nPredicted: {prediction_aggregated}")
        
        for axis in ax.flatten():
            axis.set_xticks([])
            axis.set_yticks([])
        
        for i, cam in enumerate(cams):
            # Extract the instance predictions
            prediction = predictions[i].item()
            
            # To avoid a weird overlay, threshold the CAM
            cam_mean = np.mean(cam)
            cam_std = np.std(cam)
            cam_mask = cam < cam_mean + cam_std
            cam[cam_mask] = 0
            
            # Also, we'll make the opacity a function of CAM values
            cam_min = np.min(cam)
            cam_max = np.max(cam)
            alphas = (cam - cam_min) / (2 * (cam_max - cam_min))
            alphas = np.nan_to_num(alphas)
            
            
            image = images[i, 0].cpu().numpy()
            ax[i].imshow(image, cmap="gray")
            ax[i].imshow(cam, alpha=alphas, cmap="inferno")
            ax[i].set_xlabel(f"P(Malignant) = {prediction:.2f}")
            
        return (fig, ax)