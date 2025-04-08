import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import set_seed

set_seed(seed=42)

CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
           [255, 255, 0], [255, 0, 0]]
PALETTE2CLASS = {
    (255, 255, 255): 0,  # ImSurf
    (0, 0, 255): 1,  # Building
    (0, 255, 255): 2,  # LowVeg
    (0, 255, 0): 3,  # Tree
    (255, 255, 0): 4,  # Car
    (255, 0, 0): 5  # Clutter
}
CLASS2PALETTE = {
    0: (255, 255, 255),  # ImSurf
    1: (0, 0, 255),  # Building
    2: (0, 255, 255),  # LowVeg
    3: (0, 255, 0),  # Tree
    4: (255, 255, 0),  # Car
    5: (255, 0, 0)  # Clutter
}

MEAN = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]
STD = [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]

image_only_tf_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    # # TODO test mean, std
    # transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                      std=(0.229, 0.224, 0.225)),
    transforms.Normalize(mean=MEAN, std=STD),
])

image_only_tf_val = transforms.Compose([
    transforms.ToTensor(),
    # # TODO test mean, std
    # transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                      std=(0.229, 0.224, 0.225)),
    transforms.Normalize(mean=MEAN, std=STD),
])


def rgb_to_class(mask_rgb, palette2class):
    mask_np = np.array(mask_rgb)
    h, w, _ = mask_np.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, class_idx in palette2class.items():
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


def joint_random_vertical_flip(img, mask, p=0.5):
    if np.random.rand() < p:
        return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask


def transform_train(img, mask, image_size=(512, 512)):
    img, mask = joint_resize(img, mask, size=image_size)
    img, mask = joint_random_horizontal_flip(img, mask, p=0.5)
    img, mask = joint_random_vertical_flip(img, mask, p=0.2)

    # Image-only
    img = image_only_tf_train(img)

    # Mask-only
    mask = rgb_to_class(mask, PALETTE2CLASS)
    mask = torch.from_numpy(mask).long()

    return img, mask


def transform_val(img, mask, image_size=(512, 512)):
    img, mask = joint_resize(img, mask, size=image_size)
    img = image_only_tf_val(img)
    mask = rgb_to_class(mask, PALETTE2CLASS)
    mask = torch.from_numpy(mask).long()
    return img, mask


class POTSDAMDataset(Dataset):
    def __init__(self, root_dir, mode='train', split_ratio=0.9, image_size=512, seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        self.seed = seed
        self.class_names = CLASSES
        self.num_classes = len(CLASSES)

        self.images_dir = os.path.join(root_dir, "Images")
        self.labels_dir = os.path.join(root_dir, "Labels")

        image_files = sorted(glob.glob(os.path.join(self.images_dir, "Image_*.tif")))
        paired_data = []

        for img_path in image_files:
            idx = os.path.basename(img_path).split("_")[1].split(".")[0]
            label_path = os.path.join(self.labels_dir, f"Label_{idx}.tif")
            if os.path.exists(label_path):
                paired_data.append((img_path, label_path))
            else:
                raise FileNotFoundError(f"Label not found for {img_path}")

        random.seed(seed)
        random.shuffle(paired_data)

        split_idx = int(len(paired_data) * split_ratio)
        self.pairs = paired_data[:split_idx] if mode == 'train' else paired_data[split_idx:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

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

    train_data_root = config.POTSDAM['train']
    val_data_root = config.POTSDAM['val']

    train_dataset = POTSDAMDataset(
        root_dir=train_data_root, mode='train', image_size=image_size
    )
    val_dataset = POTSDAMDataset(
        root_dir=val_data_root, mode='val', image_size=image_size
    )

    img, mask = train_dataset[0]
    show_image_and_mask(img, mask)

    img_val, mask_val = val_dataset[0]
    show_image_and_mask(img_val, mask_val)
    print("aaa")
