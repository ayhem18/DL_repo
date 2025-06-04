import os
import cv2
import torch
import numpy as np
import albumentations as A

from torch.utils.data import Dataset
from torchvision import datasets, transforms as tr
from typing import List, Tuple, Union, Optional


class MnistDSWrapper(Dataset):
    """
    Dataset wrapper for MNIST that creates label maps for conditional generation.
    
    This wrapper loads the standard MNIST dataset and creates a label map based on
    the original image shape where digit pixels are replaced with the class label.
    
    Args:
        root (str): Root directory where the dataset exists or will be downloaded to
        train (bool): If True, loads training data, otherwise loads test data
        download (bool): If True, downloads the dataset from the internet if not already downloaded
        transforms (list, optional): List of albumentations transforms to apply
        output_shape (tuple): Size to pad/resize the images to (height, width)
    """
    
    def __init__(self, 
                 root: str, 
                 train: bool = True,
                 transforms: Optional[Union[List[A.BasicTransform], A.Compose]] = None,
                 output_shape: Tuple[int, int] = (48, 48),
                 unconditional: bool = False):
        
        # use a wrapper of the mnist dataset 
        self.mnist_dataset = datasets.MNIST(
            root=root,
            train=train,
            download=not os.path.exists(root),
            transform=None  # We'll apply transforms ourselves
        )
        
        # Store parameters
        transforms = transforms or [] # default to an empty list

        # if there is no toTensor transform, add it
        if isinstance(transforms, list) and not any(isinstance(t, A.pytorch.ToTensorV2) for t in transforms):
            transforms.append(A.ToTensorV2())

        self.transforms = transforms
        self.output_shape = output_shape
        

        # Convert transforms list to Compose if needed
        if isinstance(self.transforms, List):
            # Make sure all transforms are of type A.BasicTransform
            for t in self.transforms:
                if not isinstance(t, A.BasicTransform):
                    raise TypeError(f"All transforms must be of type A.BasicTransform. Found: {type(t)}")
            
            self.transforms = A.Compose(self.transforms)

        else: 
            self.transforms = A.Compose([transforms])

        self.unconditional = unconditional


    def __len__(self):
        # return the length of the mnist dataset
        return 2000
        # return len(self.mnist_dataset)

    def _pad_to_output_shape(self, image: np.ndarray) -> np.ndarray:
        """
        Pad the image to the output shape, placing the original image in the center.
        
        Args:
            image: Input image as a numpy array
            
        Returns:
            Padded image as a numpy array
        """
        # Get current and target dimensions
        h, w = image.shape
        target_h, target_w = self.output_shape
        
        # Calculate padding
        pad_h = max(0, (target_h - h) // 2)
        pad_w = max(0, (target_w - w) // 2)
        
        # Calculate padding for each side
        pad_top = pad_h
        pad_bottom = target_h - h - pad_top
        pad_left = pad_w
        pad_right = target_w - w - pad_left
        
        # Pad the image (ensuring non-negative padding values)
        padded = np.pad(
            image,
            ((max(0, pad_top), max(0, pad_bottom)), 
             (max(0, pad_left), max(0, pad_right))),
            mode='constant',
            constant_values=0
        )
        
        # If the padded image is still not the target size, resize it
        if padded.shape != self.output_shape:
            raise ValueError(f"Padded image shape {padded.shape} does not match target shape {self.output_shape}")
            # padded = A.resize(padded, height=target_h, width=target_w)
            
        return padded

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get an item from the dataset.
        
        Returns:
            Tuple of (image, class_label, label_map)
            - image: The MNIST image as a tensor of shape (1, H, W)
            - class_label: The class label (0-9)
            - label_map: A mask where digit pixels are replaced with the class label
        """
        # Get the image and label from the MNIST dataset
        image, label = self.mnist_dataset[idx]
        
        # Convert PIL Image to numpy array
        image_np = np.array(image) / 255.0
        
        # Pad image to output shape
        padded_image = self._pad_to_output_shape(image_np)
        
        # Create a binary mask from the padded image (1 where digit, 0 elsewhere)
        binary_mask = (padded_image > 0).astype(np.float32)
        
        # Apply transformations if any
        transformed = self.transforms(image=padded_image, mask=binary_mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        
        if torch.is_tensor(transformed_mask):                            
            if transformed_mask.ndim == 3:
                # remove the first dimension if it exists
                transformed_mask = transformed_mask.squeeze(0)

            transformed_mask = transformed_mask.numpy()

        # find the bounding box of the mask
        contours, _ = cv2.findContours(transformed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            min_x, min_y, max_x, max_y = x, y, x + w, y + h
            transformed_mask[min_y:max_y, min_x:max_x] = 1

        # convert the mask to a tensor
        bounding_box_mask = torch.from_numpy(transformed_mask).to(torch.float32)

        # # Replace 1s in the mask with the class label
        # label_map = transformed_mask * label

        # Ensure image is in the right format (1, H, W)
        if len(transformed_image.shape) == 2:
            transformed_image = transformed_image.unsqueeze(0)
            
        # Ensure label map is in the right format (1, H, W)
        if len(bounding_box_mask.shape) == 2:
            bounding_box_mask = bounding_box_mask.unsqueeze(0)
        
        if self.unconditional:
            return transformed_image

        return transformed_image, label, bounding_box_mask


