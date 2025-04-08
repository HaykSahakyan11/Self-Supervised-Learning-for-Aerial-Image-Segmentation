import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import os.path as osp
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ('NoData', 'Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agricultural')
PALETTE = [
    [0, 0, 0],  # NoData (rendered as black for visualization)
    [255, 255, 255],  # Background
    [255, 0, 0],  # Building
    [255, 255, 0],  # Road
    [0, 0, 255],  # Water
    [159, 129, 183],  # Barren
    [0, 255, 0],  # Forest
    [255, 195, 128]  # Agricultural
]

PALETTE2CLASS = {
    (0, 0, 0): 0,  # NoData
    (255, 255, 255): 1,  # Background
    (255, 0, 0): 2,  # Building
    (255, 255, 0): 3,  # Road
    (0, 0, 255): 4,  # Water
    (159, 129, 183): 5,  # Barren
    (0, 255, 0): 6,  # Forest
    (255, 195, 128): 7  # Agricultural
}

CLASS2PALETTE = {
    0: (0, 0, 0),  # NoData (rendered as black for visualization)
    1: (255, 255, 255),  # Background
    2: (255, 0, 0),  # Building
    3: (255, 255, 0),  # Road
    4: (0, 0, 255),  # Water
    5: (159, 129, 183),  # Barren
    6: (0, 255, 0),  # Forest
    7: (255, 195, 128)  # Agricultural
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

image_only_tf_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

image_only_tf_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def rgb_to_class(mask_rgb, palette):
    mask_np = np.array(mask_rgb)
    h, w, _ = mask_np.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, class_idx in palette.items():
        matches = np.all(mask_np == rgb, axis=-1)
        class_mask[matches] = class_idx
    return class_mask


def class_to_rgb(mask_tensor, class2palette):
    """
    Converts class index mask (H, W) to RGB color mask (H, W, 3).
    """
    h, w = mask_tensor.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, rgb in class2palette.items():
        rgb_mask[mask_tensor == class_idx] = rgb
    return rgb_mask


def denormalize_img(img_tensor):
    """
    Denormalizes an image tensor.
    Args:
        img_tensor: A tensor of shape (C, H, W) with values normalized to [0, 1].
    Returns:
        A denormalized image as a numpy array of shape (H, W, C).
    """
    img_tensor = img_tensor.permute(1, 2, 0).numpy()  # Change to (H, W, C)
    img_tensor = img_tensor * np.array(STD) + np.array(MEAN)
    img_tensor = np.clip(img_tensor * 255, 0, 255).astype(np.uint8)
    return img_tensor


def show_image_and_mask(img_tensor, mask_tensor, class2palette=CLASS2PALETTE):
    """
    Shows image and corresponding segmentation mask side-by-side.
    """
    img_np = denormalize_img(img_tensor)
    mask_rgb = class_to_rgb(mask_tensor, class2palette)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def joint_resize(img, mask, size=(512, 512)):
    assert img.size == mask.size
    return (
        img.resize(size, resample=Image.BICUBIC),
        mask.resize(size, resample=Image.NEAREST)
    )


def joint_random_horizontal_flip(img, mask, p=0.5):
    if np.random.rand() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def joint_random_vertical_flip(img, mask, p=0.2):
    if np.random.rand() < p:
        return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask


def transform_train(img, mask, image_size=(512, 512)):
    img, mask = joint_resize(img, mask, size=image_size)
    img, mask = joint_random_horizontal_flip(img, mask, p=0.5)
    img, mask = joint_random_vertical_flip(img, mask, p=0.2)
    img = image_only_tf_train(img)
    mask = transform_mask(mask)
    return img, mask


def transform_mask(mask):
    mask_np = np.array(mask).astype(np.int64)
    return torch.from_numpy(mask_np).long()


def transform_val(img, mask, image_size=(512, 512)):
    img, mask = joint_resize(img, mask, size=image_size)
    img = image_only_tf_val(img)
    mask = transform_mask(mask)
    return img, mask


class LOVEDADataset(Dataset):
    def __init__(self, root_dir, mode='train', image_size=512):
        super().__init__()
        self.image_size = image_size
        self.mode = mode

        self.images_dir = os.path.join(root_dir, 'images_png')
        self.labels_dir = os.path.join(root_dir, 'masks_png')

        self.img_ids = sorted(os.listdir(self.images_dir))
        self.image_paths = [os.path.join(self.images_dir, img) for img in self.img_ids]
        self.mask_paths = [os.path.join(self.labels_dir, img) for img in self.img_ids]

        self.num_classes = len(CLASSES)
        self.class_names = CLASSES

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # class indices from 0–7 (0 = NoData)

        if self.mode == 'train':
            img, mask = transform_train(img, mask, image_size=(self.image_size, self.image_size))
        elif self.mode == 'val':
            img, mask = transform_val(img, mask, image_size=(self.image_size, self.image_size))
        else:
            raise ValueError("Mode must be 'train' or 'val'")

        return img, mask


if __name__ == '__main__':
    from config import CONFIG

    config = CONFIG()
    batch_size = config.batch_size  # 4
    image_size = config.image_size  # 224

    train_data_root = config.LOVEDA['train']
    val_data_root = config.LOVEDA['val']

    train_dataset = LOVEDADataset(
        root_dir=train_data_root, mode='train', image_size=image_size
    )

    val_dataset = LOVEDADataset(
        root_dir=val_data_root, mode='val', image_size=image_size
    )

    img, mask = train_dataset[0]
    show_image_and_mask(img, mask)

    img_val, mask_val = val_dataset[0]
    show_image_and_mask(img_val, mask_val)
    print("aaa")
