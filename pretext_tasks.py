import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class Mirror_Detection(object):
    def __init__(self, flip_prob=0.5, return_image=False):
        self.flip_prob = flip_prob
        self.return_image = return_image

    def __call__(self, img, alwaysFlip = False):
        if alwaysFlip or (torch.rand(1).item() < self.flip_prob):
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
        # Convert tensor to PIL Image
        img = F.to_pil_image(img)
        
        # Convert to grayscale (if not already grayscale) and expand channels to match input
        grayscale_img = F.rgb_to_grayscale(img, num_output_channels=self.num_output_channels)
        
        # Convert back to tensor if needed
        grayscale_img = F.to_tensor(grayscale_img)
        if self.return_image:
            return img, grayscale_img
        return grayscale_img

class Grayscale_Colorization_Batch(object):
    def __init__(self, return_image=False):
        self.return_image = return_image

    def __call__(self, img_batch):
        # img_batch: (B, 3, H, W)
        r, g, b = img_batch[:, 0], img_batch[:, 1], img_batch[:, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # (B, H, W)
        gray = gray.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, H, W) to match input channels

        if self.return_image:
            return img_batch, gray
        return gray

class Rotate(object):
    def __init__(self, return_image=False):
        self.angles = [0, 90, 180, 270]
        self.return_image = return_image

    def __call__(self, img):
        label = torch.randint(0, 4, (1,)).item()
        rotated_img = F.rotate(img, self.angles[label])
        if self.return_image:
            return img, rotated_img
        return rotated_img, self.angles[label]

class Jigsaw(object):
    def __init__(self, grid_size=2, permutations=None, return_image=False):
        self.grid_size = grid_size
        self.return_image = return_image
        self.permutations = permutations if permutations is not None else [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
            [2, 0, 3, 1],
        ]

    def __call__(self, img):
        # Convert tensor to PIL if needed
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)

        w, h = img.size
        tile_w, tile_h = w // self.grid_size, h // self.grid_size

        # Extract tiles
        tiles = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = j * tile_w
                upper = i * tile_h
                tile = img.crop((left, upper, left + tile_w, upper + tile_h))
                tiles.append(tile)

        # Apply random permutation
        perm_index = torch.randint(0, len(self.permutations), (1,)).item()
        perm = self.permutations[perm_index]
        shuffled_tiles = [tiles[i] for i in perm]

        # Reconstruct image
        new_img = Image.new('RGB', (w, h))
        for idx, tile in enumerate(shuffled_tiles):
            i, j = divmod(idx, self.grid_size)
            new_img.paste(tile, (j * tile_w, i * tile_h))

        if self.return_image:
            return to_tensor(img), to_tensor(new_img)
        return to_tensor(new_img), perm_index
    
class Jigsaw_Batch(object):
    def __init__(self, grid_size=2, permutations=None, return_image=False):
        self.jigsaw = Jigsaw(grid_size=grid_size,
                             permutations=permutations,
                             return_image=return_image)

    def __call__(self, batch):
        origs, shuffled = [], []
        for img in batch:
            o, s = self.jigsaw(img)        # apply single-image Jigsaw
            origs.append(o)
            shuffled.append(s)
        # stack back into (B, C, H, W) tensors
        return torch.stack(origs), torch.stack(shuffled)