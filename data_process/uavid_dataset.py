import os
import json
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import set_seed, CONFIG

config = CONFIG()
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

orig_H = 2160
orig_W = 3840

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


class PatchedUAVIDDataset(UAVIDDataset):
    def __init__(self, split: str, img_size: int, data_root=None):
        if data_root is None:
            raise ValueError("data_root must be specified for PatchedUAVIDDataset")
        super().__init__(data_root=data_root,
                         img_dir='images', mask_dir='labels',
                         mode='val',
                         img_suffix='.png', mask_suffix='.png',
                         image_size=img_size,
                         transform=None, target_transform=None)

    # ↓ add the filename (without extension) to the returned tuple
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        base = os.path.splitext(os.path.basename(self.img_ids[idx]))[0]
        return img, base


class UAVIDPatchStitch(Dataset):
    """
    Stitches arbitrary patch grids back into one canvas.

    Parameters
    ----------
    split : 'train' | 'val' | 'test'
    rep   : 'logits' | 'argmax'
    resize: (h, w) or None
    expected_n : int or None
        If given, keep only groups that contain exactly this number of patches
        (4, 9, 60, 144, …).  If None, **any** complete group is kept.
    """

    def __init__(
            self,
            split="train",
            rep="logits",
            resize=None,
            patch_meta=None,
            logits_root=None,
            label_root=None,
            expected_n: int | None = None,
    ):
        assert rep in {"logits", "argmax"}
        self.rep = rep
        self.resize = resize
        self.N_req = expected_n

        self.patch_meta = patch_meta or os.path.join(
            config.UAVID_patched['9']['no_overlap'][split], "patches_metadata.json"
        )
        self.logits_root = logits_root or config.UAVID_patch_inf['dino_mc']['9']['no_overlap'][split]
        self.label_root = label_root or os.path.join(config.UAVID[split], "labels")

        self.groups = self._collect_groups()
        any_patch = next(iter(next(iter(self.groups.values()))))[0]
        self.C = np.load(os.path.join(self.logits_root,
                                      any_patch.replace(".png", ".npy"))).shape[0]
        self.class_names = CLASSES

    # ----------------------------------------------------------------------
    def _collect_groups(self):
        with open(self.patch_meta) as f:
            meta = json.load(f)

        groups = {}
        for p_name, coord in meta.items():
            base = p_name.rsplit("_", 1)[0]  # image_x_y.png → image
            groups.setdefault(base, []).append((p_name, coord))

        complete = {}
        for base, lst in groups.items():
            # filter out missing .npy
            if all(os.path.isfile(os.path.join(self.logits_root,
                                               p[0].replace(".png", ".npy"))) for p in lst):
                if self.N_req is None or len(lst) == self.N_req:
                    # row-major sort
                    lst.sort(key=lambda t: (t[1]["y_start"], t[1]["x_start"]))
                    complete[base] = lst

        if not complete:
            msg = "No groups match "
            msg += f"N={self.N_req}" if self.N_req else "metadata"
            raise RuntimeError(msg + f" under {self.logits_root}")
        return complete

    # helpers ---------------------------------------------------------------
    @staticmethod
    def _extent(coords):
        return max(c["y_end"] for c in coords), max(c["x_end"] for c in coords)

    # PyTorch API -----------------------------------------------------------
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        base, patches = list(self.groups.items())[idx]
        coords_only = [c for _, c in patches]
        H_big, W_big = self._extent(coords_only)

        # canvas ───────────────────────────────────────────────────────────
        if self.rep == "logits":
            canvas = torch.zeros(self.C, H_big, W_big, dtype=torch.float32)
        else:
            canvas = torch.zeros(H_big, W_big, dtype=torch.int64)

        for p_name, c in patches:
            logit = torch.from_numpy(
                np.load(os.path.join(self.logits_root, p_name.replace(".png", ".npy")))
            )  # (C,h,w)

            h_exp, w_exp = c["y_end"] - c["y_start"], c["x_end"] - c["x_start"]
            if logit.shape[1:] != (h_exp, w_exp):
                logit = F.interpolate(logit[None], size=(h_exp, w_exp),
                                      mode="bicubic", align_corners=False).squeeze(0)

            if self.rep == "logits":
                canvas[:, c["y_start"]:c["y_end"], c["x_start"]:c["x_end"]] = logit
            else:
                canvas[c["y_start"]:c["y_end"], c["x_start"]:c["x_end"]] = logit.argmax(0)

        # full-resolution ground truth ─────────────────────────────────────
        gt_rgb = Image.open(os.path.join(self.label_root, base + ".png")).convert("RGB")
        gt = torch.from_numpy(rgb_to_class(gt_rgb, PALETTE2CLASS)).long()

        # optional resize ──────────────────────────────────────────────────
        if self.resize:
            h, w = self.resize
            if self.rep == "logits":
                canvas = F.interpolate(canvas[None], size=(h, w),
                                       mode="bicubic", align_corners=False).squeeze(0)
            else:
                canvas = F.interpolate(canvas[None, None].float(), size=(h, w),
                                       mode="bicubic").squeeze().long()
            gt = F.interpolate(gt[None, None].float(), size=(h, w),
                               mode="nearest").squeeze().long()

        return canvas, gt


def tensor_to_rgb(mask: torch.Tensor) -> np.ndarray:
    """
    Convert a (H,W) class–index tensor to an RGB uint8 image using UAVID palette.
    """
    mask_np = mask.cpu().numpy()
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS2PALETTE.items():
        rgb[mask_np == cls] = color  # color = (R,G,B)
    return rgb


def show_patch_aggregation(logits: torch.Tensor, gt: torch.Tensor, title=''):
    """
    Visualise stitched logits vs. ground-truth mask.
      • logits … (C,H,W)  OR  (H,W) argmax mask
      • gt     … (H,W)    class indices
    """
    if logits.ndim == 3:  # convert logits → predicted mask
        pred = logits.argmax(0)
    else:
        pred = logits.clone()

    pred_rgb = tensor_to_rgb(pred)
    gt_rgb = tensor_to_rgb(gt)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(pred_rgb);
    ax[0].set_title('Predicted');
    ax[0].axis('off')
    ax[1].imshow(gt_rgb);
    ax[1].set_title('Ground truth');
    ax[1].axis('off')
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout();
    plt.show()


if __name__ == '__main__':
    # choose a grid size – here 3×3 patches with 10 % overlap
    patch_meta = config.UAVID_patched['9']['no_overlap']['train'] + '/patches_metadata.json'
    logits_root = config.UAVID_patch_inf['dino_mc']['9']['no_overlap']['train']
    label_root = config.UAVID['train'] + '/labels'

    dataset_ = UAVIDPatchStitch(split="train",
                                patch_meta=patch_meta,
                                logits_root=logits_root,
                                label_root=label_root,
                                expected_n=9,  # force exactly 9 patches
                                rep="logits",  # logits or argmax
                                resize=None)  # resize=(512, 512)
    print(len(dataset_), dataset_[0][0].shape, dataset_[0][1].shape)
    show_patch_aggregation(dataset_[0][0], dataset_[0][1])

    # Example usage: UAVIDDataset
    # from config import CONFIG
    #
    # config = CONFIG()
    # batch_size = config.batch_size  # 4
    # image_size = config.image_size  # 224
    #
    # train_data_root = config.UAVID['train']
    # val_data_root = config.UAVID['val']
    #
    # train_dataset = UAVIDDataset(
    #     data_root=train_data_root, img_dir='images', mask_dir='labels', mode='train',
    #     img_suffix='.png', mask_suffix='.png',
    #     image_size=224, transform=None, target_transform=None
    # )
    #
    # val_dataset = UAVIDDataset(
    #     data_root=val_data_root, img_dir='images', mask_dir='labels', mode='val',
    #     img_suffix='.png', mask_suffix='.png',
    #     image_size=224, transform=None, target_transform=None
    # )
    #
    # img, mask = train_dataset[0]
    # show_image_and_mask(img, mask)
    #
    # img_val, mask_val = val_dataset[0]
    # show_image_and_mask(img_val, mask_val)
    # print("aaa")
