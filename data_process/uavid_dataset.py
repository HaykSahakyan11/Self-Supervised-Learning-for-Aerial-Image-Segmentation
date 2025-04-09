import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import set_seed

set_seed(seed=42)

CLASSES = ('Clutter', 'Building', 'Road', 'Tree', 'LowVeg', 'Moving_Car', 'Static_Car', 'Human')
PALETTE = [[0, 0, 0], [128, 0, 0], [128, 64, 128], [0, 128, 0], [128, 128, 0], [64, 0, 128], [192, 0, 192], [64, 64, 0]]
PALETTE2CLASS = {
    (0, 0, 0): 0,  # Clutter
    (128, 0, 0): 1,  # Building
    (128, 64, 128): 2,  # Road
    (0, 128, 0): 3,  # Tree
    (128, 128, 0): 4,  # Low Vegetation
    (64, 0, 128): 5,  # Moving Car
    (192, 0, 192): 6,  # Static Car
    (64, 64, 0): 7  # Human
}
CLASS2PALETTE = {
    0: (0, 0, 0),  # Clutter
    1: (128, 0, 0),  # Building
    2: (128, 64, 128),  # Road
    3: (0, 128, 0),  # Tree
    4: (128, 128, 0),  # Low Vegetation
    5: (64, 0, 128),  # Moving Car
    6: (192, 0, 192),  # Static Car
    7: (64, 64, 0)  # Human
}

MEAN = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]
STD = [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]

image_only_tf_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

image_only_tf_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def rgb_to_class(mask_rgb, palette2class):
    mask = np.array(mask_rgb)
    h, w, _ = mask.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, class_idx in palette2class.items():
        matches = np.all(mask == rgb, axis=-1)
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
    # Joint spatial transforms
    img, mask = joint_resize(img, mask, size=image_size)
    img, mask = joint_random_horizontal_flip(img, mask, p=0.5)
    img, mask = joint_random_vertical_flip(img, mask, p=0.2)

    # Image-only
    img = image_only_tf_train(img)

    # Mask-only
    mask = transform_mask(mask)

    return img, mask

def transform_mask(mask):
    # Mask-only
    mask = rgb_to_class(mask, PALETTE2CLASS)
    mask = torch.from_numpy(mask).long()
    return mask


def transform_val(img, mask, image_size=(512, 512)):
    img, mask = joint_resize(img, mask, size=image_size)

    img = image_only_tf_val(img)
    mask = transform_mask(mask)
    return img, mask


def transform_test(img, image_size=(512, 512)):
    img = img.resize(image_size, resample=Image.BICUBIC)
    img = image_only_tf_val(img)
    return img


class UAVIDDataset(Dataset):
    def __init__(
            self, data_root, img_dir='images', mask_dir='labels', mode='train',
            img_suffix='.png', mask_suffix='.png',
            image_size=224, transform=None, target_transform=None
    ):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.mode = mode

        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = len(CLASSES)
        self.class_names = CLASSES
        self.palette = PALETTE
        self.palette2class = PALETTE2CLASS
        self.class2palette = CLASS2PALETTE

        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

        self.image_paths = []
        self.mask_paths = []

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img, mask = self.load_img_and_mask(index)

        # Resize both before conversion
        img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        if self.mode == 'train':
            img, mask = transform_train(img, mask, image_size=(self.image_size, self.image_size))
        elif self.mode == 'val':
            img, mask = transform_val(img, mask, image_size=(self.image_size, self.image_size))
        elif self.mode == 'test':
            img = transform_test(img, image_size=(self.image_size, self.image_size))
            return img
        else:
            raise ValueError("Invalid mode. Choose 'train' or 'val'.")

        return img, mask

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('RGB')
        return img, mask


if __name__ == '__main__':
    from config import CONFIG

    config = CONFIG()
    batch_size = config.batch_size  # 4
    image_size = config.image_size  # 224

    train_data_root = config.UAVID['train']
    val_data_root = config.UAVID['val']

    train_dataset = UAVIDDataset(
        data_root=train_data_root, img_dir='images', mask_dir='labels', mode='train',
        img_suffix='.png', mask_suffix='.png',
        image_size=224, transform=None, target_transform=None
    )

    val_dataset = UAVIDDataset(
        data_root=val_data_root, img_dir='images', mask_dir='labels', mode='val',
        img_suffix='.png', mask_suffix='.png',
        image_size=224, transform=None, target_transform=None
    )

    img, mask = train_dataset[0]
    show_image_and_mask(img, mask)

    img_val, mask_val = val_dataset[0]
    show_image_and_mask(img_val, mask_val)
    print("aaa")
