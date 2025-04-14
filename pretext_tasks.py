import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class Mirror_Detection(object):
    def __init__(self, flip_prob=0.5, return_image=False):
        self.flip_prob = flip_prob
        self.return_image = return_image

    def __call__(self, img):
        if torch.rand(1).item() < self.flip_prob:
            flipped = F.vflip(img)
            label = 1  # flipped
        else:
            flipped = img
            label = 0  # original

        if self.return_image:
            return img, flipped
        else:
            return flipped


class Grayscale_Colorization(object):
    def __init__(self, num_output_channels=1, return_image=False):
        """
        num_output_channels: Number of output channels (1 for grayscale, 3 for 3-channel grayscale).
        return_image: If True, returns a tuple (original_image, grayscale_image).
        """
        self.num_output_channels = num_output_channels
        self.return_image = return_image

    def __call__(self, img):
        grayscale_img = F.to_grayscale(img, num_output_channels=self.num_output_channels)
        if self.return_image:
            return img, grayscale_img
        return grayscale_img

class Rotate_Jigsaw(object):
    def __init__(self, n_patches=(3, 3), num_rotations=4, return_info=False):
        """
        n_patches: Tuple indicating the grid size for jigsaw (e.g., 3x3).
        num_rotations: Number of discrete rotations (default is 4 for 0°, 90°, 180°, 270°).
        return_info: If True, also returns the rotation label and jigsaw permutation index.
        """
        self.n_patches = n_patches
        self.degrees = torch.arange(num_rotations) * (360.0 / num_rotations)
        self.return_info = return_info

    def __call__(self, img):
        assert isinstance(img, torch.Tensor), "Input should be a torch.Tensor (e.g., from ToTensor())"

        # Ensure img is 4D (batch_size, C, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)  # Add batch dimension if it's a single image

        # Rotation
        rot_label = torch.randint(len(self.degrees), (1,)).item()
        rotated_img = F.rotate(img, angle=self.degrees[rot_label].item())

        # Jigsaw
        patch_size_1 = rotated_img.size(1) // self.n_patches[0]  # Height of each patch
        patch_size_2 = rotated_img.size(2) // self.n_patches[1]  # Width of each patch

        patches = rotated_img.unfold(1, patch_size_1, patch_size_1).unfold(2, patch_size_2, patch_size_2)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # Corrected permute
        patches = patches.view(-1, patches.shape[3], patches.shape[4], patches.shape[5])  # (num_patches, C, H, W)

        rand_perm = torch.randperm(patches.shape[0])
        shuffled_patches = patches[rand_perm]

        # Optional: return additional info
        if self.return_info:
            return img, shuffled_patches, rot_label, rand_perm
        return img, shuffled_patches