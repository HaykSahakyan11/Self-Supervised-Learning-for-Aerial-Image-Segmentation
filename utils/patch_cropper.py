# patch_cropper.py
"""
Utilities for cutting a large (image, mask) pair into a regular grid of
patches *with optional overlap* and writing both the patches and their
coordinates into a metadata‑dict that you can later dump to JSON.

Example
-------
>>> meta = {}
>>> split_and_save_image(
...     image_path="seq5_000500.png",
...     mask_path ="seq5_000500_mask.png",
...     save_img_dir="uavid_patched/train/images",
...     save_msk_dir="uavid_patched/train/labels",
...     meta_dict   = meta,
...     grid        = 9,         # 3 × 3
...     overlap_ratio = 0.15     # 15 % overlap
... )
>>> json.dump(meta, open("patches_metadata.json","w"), indent=2)
"""
import cv2
import math, os, json

# from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Union

from config import CONFIG, set_seed

config = CONFIG()


def _grid_from_arg(grid: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert 4→(2,2) ; 9→(3,3) ; validate perfect square if int."""
    if isinstance(grid, int):
        root = int(math.sqrt(grid))
        if root * root != grid:
            raise ValueError("grid=int must be a perfect square: 4, 9, 16 …")
        return root, root
    rows, cols = grid
    if rows <= 0 or cols <= 0:
        raise ValueError("grid tuple must contain positive integers")
    return rows, cols


def split_and_save_image(
        image_path: str,
        mask_path: str,
        save_img_dir: str,
        save_msk_dir: str,
        meta_dict: Dict[str, Dict],
        *,
        grid: Union[int, Tuple[int, int]] = 4,  # 2×2 by default
        overlap_ratio: float | None = None,  # 0 ≤ o < 1 ; None → 0
        filename_fmt: str = "{base}_{row}{col}.png",
) -> List[str]:
    """
    Cut one (image, mask) pair into an R×C grid of patches.

    Parameters
    ----------
    image_path / mask_path : str
        Source files (RGB or BGR are fine – we don't modify pixels).
    save_img_dir / save_msk_dir : str
        Output directories – created if missing.
    meta_dict : dict
        Will be **updated** with an entry for every patch:
        ``meta_dict[filename] = {'x_start':..,'y_start':..,'x_end':..,'y_end':..}``
    grid : int or (rows, cols)
        `4` (=2×2), `9` (=3×3) …  or an explicit tuple.
    overlap_ratio : float, optional
        0 → no overlap.  0.15 → patches are 15 % larger than their stride
        in *both* directions.  Must satisfy 0 ≤ r < 1.
    filename_fmt : str
        Format string; receives `base`, `row`, `col` kwargs.

    Returns
    -------
    List[str]
        The list of **patch filenames** that were written (image & mask
        share the same name).
    """
    if overlap_ratio is None:
        overlap_ratio = 0.0
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1)")

    rows, cols = _grid_from_arg(grid)

    # make sure output dirs exist
    Path(save_img_dir).mkdir(parents=True, exist_ok=True)
    Path(save_msk_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # read original data
    # ------------------------------------------------------------------
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    msk = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if img is None or msk is None:
        raise FileNotFoundError("Could not read image or mask")
    assert img.shape[:2] == msk.shape[:2], "image & mask size mismatch"
    H, W = img.shape[:2]

    # ------------------------------------------------------------------
    # compute patch geometry
    # ------------------------------------------------------------------
    stride_x = W / cols
    stride_y = H / rows
    patch_w = int(round(stride_x * (1 + overlap_ratio)))
    patch_h = int(round(stride_y * (1 + overlap_ratio)))
    step_x = int(round(stride_x * (1 - overlap_ratio)))
    step_y = int(round(stride_y * (1 - overlap_ratio)))

    base = Path(image_path).stem  # e.g. seq5_000500
    written: List[str] = []

    for r in range(rows):
        for c in range(cols):
            x0 = min(c * step_x, W - patch_w)
            y0 = min(r * step_y, H - patch_h)
            x1, y1 = x0 + patch_w, y0 + patch_h

            img_patch = img[y0:y1, x0:x1]
            msk_patch = msk[y0:y1, x0:x1]

            fname = filename_fmt.format(base=base, row=r, col=c)
            cv2.imwrite(os.path.join(save_img_dir, fname), img_patch)
            cv2.imwrite(os.path.join(save_msk_dir, fname), msk_patch)
            written.append(fname)

            meta_dict[fname] = dict(
                x_start=int(x0), y_start=int(y0),
                x_end=int(x1), y_end=int(y1)
            )

    return written


def split_image_and_mask_custom_grid(
        img_path: str,
        mask_path: str,
        img_save_dir: str,
        mask_save_dir: str,
        H_sep_num: int,
        W_sep_num: int,
        filename_fmt: str = "{base}_{row}_{col}.png",
        meta_dict: Dict[str, Dict] = None,
) -> List[str]:
    Path(img_save_dir).mkdir(parents=True, exist_ok=True)
    Path(mask_save_dir).mkdir(parents=True, exist_ok=True)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

    if img is None or mask is None:
        raise FileNotFoundError(f"Could not read image or mask: {img_path}, {mask_path}")

    assert img.shape[:2] == mask.shape[:2], "Image and mask dimensions must match"

    H, W = img.shape[:2]

    patch_h, patch_w = H // H_sep_num, W // W_sep_num

    base = Path(img_path).stem
    written = []

    for row in range(H_sep_num):
        for col in range(W_sep_num):
            y0, y1 = row * patch_h, (row + 1) * patch_h
            x0, x1 = col * patch_w, (col + 1) * patch_w

            img_patch = img[y0:y1, x0:x1]
            mask_patch = mask[y0:y1, x0:x1]

            fname = filename_fmt.format(base=base, row=row, col=col)
            cv2.imwrite(os.path.join(img_save_dir, fname), img_patch)
            cv2.imwrite(os.path.join(mask_save_dir, fname), mask_patch)
            written.append(fname)

            if meta_dict is not None:
                meta_dict[fname] = dict(
                    x_start=x0, y_start=y0,
                    x_end=x1, y_end=y1
                )

    return written


if __name__ == '__main__':
    meta = {}

    mode = "train"
    # mode = "val"
    base_path = config.base_path
    overlap = 0.0  # 10 % overlap – set to 0 for none

    # For grid patching grid = m = n × n - split_and_save_image
    grid = 4  # 2×2 patches, 9 for 3×3 patches

    H_sep_num = 3  # 9
    W_sep_num = 4  # 16
    orig_H = 2160
    orig_W = 3840
    out_H = orig_H // H_sep_num
    out_W = orig_W // W_sep_num

    # dataset UAVID

    img_dir = os.path.join(base_path, f"datasets/UAVID/{mode}/images/")
    msk_dir = os.path.join(base_path, f"datasets/UAVID/{mode}/labels/")

    # For grid patching grid = m = n × n - split_and_save_image

    # out_img = os.path.join(base_path, f"datasets/UAVID_patched_{grid}/{mode}/images")
    # out_msk = os.path.join(base_path, f"datasets/UAVID_patched_{grid}/{mode}/labels")

    # For custom grid patching grid = m × n - split_image_and_mask_custom_grid
    out_img = os.path.join(base_path,
                           f"datasets/UAVID_patched_{out_H}_{out_W}_count_{H_sep_num * W_sep_num}/{mode}/images")
    out_msk = os.path.join(base_path,
                           f"datasets/UAVID_patched_{out_H}_{out_W}_count_{H_sep_num * W_sep_num}/{mode}/labels")


    # dataset UDD6

    # img_dir = os.path.join(base_path, f"datasets/UDD6/{mode}/src/")
    # msk_dir = os.path.join(base_path, f"datasets/UDD6/{mode}/gt/")
    #
    # # 1 For grid patching grid = m = n × n - split_and_save_image
    # out_img = os.path.join(base_path, f"datasets/UDD6_patched_{grid}/{mode}/src")
    # out_msk = os.path.join(base_path, f"datasets/UDD6_patched_{grid}/{mode}/gt")

    # 2 For custom grid patching grid = m × n - split_image_and_mask_custom_grid

    # out_img = os.path.join(base_path,
    #                        f"datasets/UDD6_patched_{out_H}_{out_W}_count_{H_sep_num * W_sep_num}_/{mode}/src")
    # out_msk = os.path.join(base_path,
    #                        f"datasets/UDD6_patched_{out_H}_{out_W}_count_{H_sep_num * W_sep_num}/{mode}/gt")

    # for fname in os.listdir(img_dir):
    #     split_and_save_image(
    #         image_path=os.path.join(img_dir, fname),
    #         mask_path=os.path.join(msk_dir, fname),
    #         save_img_dir=out_img,
    #         save_msk_dir=out_msk,
    #         meta_dict=meta,
    #         grid=grid,
    #         overlap_ratio=overlap
    #     )

    for fname in os.listdir(img_dir):
        split_image_and_mask_custom_grid(
            img_path=os.path.join(img_dir, fname),
            mask_path=os.path.join(msk_dir, fname),
            img_save_dir=out_img,
            mask_save_dir=out_msk,
            H_sep_num=H_sep_num,
            W_sep_num=W_sep_num,
            meta_dict=meta
        )

    # UAVID
    # For grid patching grid = m = n × n - split_and_save_image
    # metadata_path = os.path.join(base_path,
    #                              f"datasets/UAVID_patched_{grid}/{mode}/patches_metadata.json")
    metadata_path = os.path.join(base_path,
                                 f"datasets/UAVID_patched_{out_H}_{out_W}_count_{H_sep_num * W_sep_num}/{mode}/patches_metadata.json")

    # UDD6
    # For grid patching grid = m = n × n - split_and_save_image
    # metadata_path = os.path.join(base_path,
    #                              f"datasets/UDD6_patched_{grid}/{mode}/patches_metadata.json")
    # metadata_path = os.path.join(base_path,
    #                              f"datasets/UDD6_patched_{out_H}_{out_W}_count_{H_sep_num * W_sep_num}/{mode}/patches_metadata.json")

    with open(metadata_path, "w") as fp:
        json.dump(meta, fp, indent=2)
