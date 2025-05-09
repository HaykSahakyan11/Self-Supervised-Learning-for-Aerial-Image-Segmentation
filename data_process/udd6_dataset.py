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
from config import set_seed, CONFIG
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
from pathlib import Path


config = CONFIG()
set_seed(seed=42)

CLASSES = ('Other', 'Facade', 'Road', 'Vegetation', 'Vehicle', 'Roof')
PALETTE = [
    [0, 0, 0],  # Other
    [102, 102, 156],  # Facade
    [128, 64, 128],  # Road
    [107, 142, 35],  # Vegetation
    [0, 0, 142],  # Vehicle
    [70, 70, 70]  # Roof
]
PALETTE2CLASS = {tuple(palet): i for i, palet in enumerate(PALETTE)}
CLASS2PALETTE = {i: tuple(palet) for i, palet in enumerate(PALETTE)}

MEAN = [0.3918, 0.4114, 0.3726]
STD = [0.1553, 0.1528, 0.1456]

SRC_SUFFIX = '.JPG'
GT_SUFFIX = '.png'

_affine_aug = A.OneOf([
    # TODO update angle -90 90
    A.Affine(rotate=(-90, 90), p=0.7),
    A.Affine(translate_percent={"x": (-0.05, 0.05),
                                "y": (-0.05, 0.05)}, p=0.2),
    A.Affine(scale=(0.9, 1.1), p=0.5),
    A.Affine(shear=(-5, 5), p=0.5)
], p=0.9)  # 80 % chance to pick ONE of the 4

image_only_tf_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

image_only_tf_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


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


def joint_afine_transform(img, mask):
    # convert PIL → numpy for Albumentations
    img_np = np.array(img)
    mask_np = np.array(mask)

    # Apply affine transformations to both image and mask
    transform = _affine_aug(image=img_np, mask=mask_np)
    img_np = transform['image']
    mask_np = transform['mask']

    # convert back to PIL
    img = Image.fromarray(img_np)
    mask = Image.fromarray(mask_np)
    return img, mask


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


def transform_train(img, mask, image_size=(512, 512)):
    # Joint spatial transforms
    img, mask = joint_resize(img, mask, size=image_size)
    # TODO: add affine transform and check parameters
    img, mask = joint_random_horizontal_flip(img, mask, p=0.5)
    img, mask = joint_random_vertical_flip(img, mask, p=0.5)
    img, mask = joint_afine_transform(img, mask)

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


class UDD6Dataset(Dataset):
    def __init__(
            self, data_root, img_dir='src', mask_dir='gt',
            use_metadata=False, metadata_path=None, image_size=224, mode='val',
            img_suffix=None, mask_suffix=None,
    ):
        self.data_root = data_root
        self.mode = mode
        self.image_size = image_size

        self.num_classes = len(CLASSES)
        self.class_names = CLASSES
        self.palette = PALETTE
        self.palette2class = PALETTE2CLASS
        self.class2palette = CLASS2PALETTE

        self.img_dir = f'{self.data_root}/{img_dir}'
        self.mask_dir = f'{self.data_root}/{mask_dir}'
        self.metadata_file = f'{metadata_path}/{mode}.txt'

        self.img_suffix = img_suffix if img_suffix else SRC_SUFFIX
        self.mask_suffix = mask_suffix if mask_suffix else GT_SUFFIX

        if use_metadata:
            assert metadata_path is not None
            with open(self.metadata_file, 'r') as f:
                # TODO not implemented
                stems = [ln.strip() for ln in f if ln.strip()]
                raise NotImplementedError('Please set use_metadata=False')
        else:
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


class PatchedUDD6Dataset(UDD6Dataset):
    def __init__(self, split: str, img_size: int, data_root=None):
        if data_root is None:
            raise ValueError("data_root must be specified for PatchedUDD6Dataset")
        super().__init__(data_root=data_root,
                         img_dir='src', mask_dir='gt',
                         mode='val',
                         image_size=img_size,
                         )

    # ↓ add the filename (without extension) to the returned tuple
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        base = os.path.splitext(os.path.basename(self.img_ids[idx]))[0]
        return img, base

class UDD6PatchStitch(Dataset):
    """
    Re‑assemble predicted patch logits (or argmax masks) into the original
    full‑resolution UDD6 canvas and return the matching GT.

    Parameters
    ----------
    split        : 'train' | 'val' | 'test'
    rep          : 'logits' | 'argmax'
    resize       : (H, W) or None – optional final resizing
    patch_meta   : JSON with coords for every patch
    logits_root  : dir that holds <patch_stem>.npy  (or .pt / .bin / .py)
    label_root   : dir with original RGB GT masks
    expected_n   : keep only images that have exactly this many patches
    """

    _LOGIT_EXTS = (".npy", ".pt", ".bin", ".py")  # extend if you use others

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        split: str = "val",
        rep: str = "logits",
        resize: Optional[Tuple[int, int]] = None,
        patch_meta: Optional[str | Path] = None,
        logits_root: Optional[str | Path] = None,
        label_root: Optional[str | Path] = None,
        expected_n: Optional[int] = None,
    ):
        assert rep in {"logits", "argmax"}, "rep must be 'logits' or 'argmax'"
        self.rep, self.resize, self.N_req = rep, resize, expected_n

        self.patch_meta = (
            Path(patch_meta)
            if patch_meta is not None
            else Path(config.UDD6_patched[split]) / "patches_metadata.json"
        )
        self.logits_root = (
            Path(logits_root)
            if logits_root is not None
            else Path(config.UDD6_patch_inf["dino_deit"][split])
        )
        self.label_root = (
            Path(label_root)
            if label_root is not None
            else Path(config.UDD6[split]) / "gt"
        )

        self.groups = self._collect_groups()
        # infer #classes
        some_stem = next(iter(next(iter(self.groups.values()))))[0]
        self.C = self._load_logit(some_stem).shape[0]  # (C,h,w)

    # ------------------------------------------------------------------ #
    def _logit_exists(self, stem: str) -> bool:
        """True if any allowed extension file exists for <stem>."""
        for ext in self._LOGIT_EXTS:
            if (self.logits_root / f"{stem}{ext}").is_file():
                return True
        return False

    def _load_logit(self, stem: str) -> np.ndarray:
        """Load logits/argmax for <stem> regardless of extension."""
        for ext in self._LOGIT_EXTS:
            f = self.logits_root / f"{stem}{ext}"
            if f.is_file():
                if ext == ".pt":
                    return torch.load(f, map_location="cpu").numpy()
                return np.load(f, allow_pickle=True)
        raise FileNotFoundError(f"No logit file found for patch stem '{stem}'")

    # ------------------------------------------------------------------ #
    def _collect_groups(self) -> Dict[str, List[Tuple[str, Dict]]]:
        """
        Returns
        -------
        groups : {base_img : [(patch_stem, coord_dict), ...]}
        """
        with open(self.patch_meta) as f:
            meta: Dict[str, Dict] = json.load(f)

        groups: Dict[str, List[Tuple[str, Dict]]] = {}
        for p_name, coord in meta.items():
            stem = p_name.rsplit(".", 1)[0]            # strip ext if present
            base = stem.rsplit("_", 2)[0]              # img_x_y → img
            groups.setdefault(base, []).append((stem, coord))

        complete: Dict[str, List[Tuple[str, Dict]]] = {}
        for base, lst in groups.items():
            if all(self._logit_exists(stem) for stem, _ in lst):
                if self.N_req is None or len(lst) == self.N_req:
                    lst.sort(key=lambda t: (t[1]["y_start"], t[1]["x_start"]))
                    complete[base] = lst

        if not complete:
            msg = "No patch groups found "
            msg += f"with N={self.N_req} " if self.N_req else ""
            raise RuntimeError(msg + f"in {self.logits_root}")
        return complete

    @staticmethod
    def _extent(coords: List[Dict]) -> Tuple[int, int]:
        return (max(c["y_end"] for c in coords),
                max(c["x_end"] for c in coords))

    # --------------------- PyTorch Dataset API ------------------------ #
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: int):
        base, patches = list(self.groups.items())[idx]
        coords_only = [c for _, c in patches]
        H_big, W_big = self._extent(coords_only)

        canvas = (
            torch.zeros(self.C, H_big, W_big, dtype=torch.float32)
            if self.rep == "logits"
            else torch.zeros(H_big, W_big, dtype=torch.int64)
        )

        for stem, c in patches:
            logit = torch.from_numpy(self._load_logit(stem))  # (C,h,w) or (h,w)

            # adjust if split rounding produced size mismatch
            h_exp, w_exp = c["y_end"] - c["y_start"], c["x_end"] - c["x_start"]
            if logit.ndim == 3 and logit.shape[1:] != (h_exp, w_exp):
                logit = F.interpolate(logit[None], size=(h_exp, w_exp),
                                      mode="bicubic", align_corners=False).squeeze(0)
            elif logit.ndim == 2 and logit.shape != (h_exp, w_exp):
                logit = F.interpolate(logit[None, None].float(),
                                      size=(h_exp, w_exp),
                                      mode="nearest").squeeze().long()

            if self.rep == "logits":
                canvas[:, c["y_start"]:c["y_end"], c["x_start"]:c["x_end"]] = logit
            else:
                canvas[c["y_start"]:c["y_end"], c["x_start"]:c["x_end"]] = (
                    logit if logit.ndim == 2 else logit.argmax(0)
                )

        # -------- load GT mask ---------------------------------------
        gt_rgb = Image.open(self.label_root / f"{base}.png").convert("RGB")
        gt = torch.from_numpy(rgb_to_class(gt_rgb, PALETTE2CLASS)).long()

        # -------- optional resize ------------------------------------
        if self.resize is not None:
            h_tgt, w_tgt = self.resize
            if self.rep == "logits":
                canvas = F.interpolate(canvas[None], size=(h_tgt, w_tgt),
                                       mode="bicubic",
                                       align_corners=False).squeeze(0)
            else:
                canvas = F.interpolate(canvas[None, None].float(),
                                       size=(h_tgt, w_tgt),
                                       mode="bicubic").squeeze().long()
            gt = F.interpolate(gt[None, None].float(),
                               size=(h_tgt, w_tgt),
                               mode="nearest").squeeze().long()

        return canvas, gt


# ---------------- convenience visualiser ------------------------------ #
def tensor_to_rgb(mask: torch.Tensor) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, colour in CLASS2PALETTE.items():
        rgb[mask == cls] = colour
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
    ax[0].imshow(pred_rgb)
    ax[0].set_title('Predicted')
    ax[0].axis('off')
    ax[1].imshow(gt_rgb)
    ax[1].set_title('Ground truth')
    ax[1].axis('off')
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage
    # image_size = config.image_size  # 224
    #
    # train_data_root = config.UDD6['train']
    # val_data_root = config.UDD6['val']
    # metadata_path = config.UDD6['metadata']
    #
    # train_dataset = UDD6Dataset(
    #     data_root=train_data_root, img_dir='src', mask_dir='gt',
    #     metadata_path=metadata_path,
    #     image_size=224, mode='val', use_metadata=False,
    # )
    #
    # img, mask = train_dataset[0]
    # show_image_and_mask(img, mask)

    # choose a grid size – here 3×3 patches
    patch_meta = config.UDD6_patched['4']['val'] + '/patches_metadata.json'
    logits_root = config.UDD6_patch_inf['dino_deit']['4']['val']
    label_root = config.UDD6['val'] + '/gt'

    dataset_ = UDD6PatchStitch(split="val",
                                patch_meta=patch_meta,
                                logits_root=logits_root,
                                label_root=label_root,
                                expected_n=4,  # force exactly 9 patches
                                rep="logits",  # logits or argmax
                                resize=None)  # resize=(512, 512)
    print(len(dataset_), dataset_[0][0].shape, dataset_[0][1].shape)
    show_patch_aggregation(dataset_[0][0], dataset_[0][1])
    show_patch_aggregation(dataset_[1][0], dataset_[1][1])
