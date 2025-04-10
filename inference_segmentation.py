import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings
import albumentations as A

from PIL import Image

from models.networks import UPerNetDinoMCViT
from data_process import uavid_dataset, potsdam_dataset, loveda_dataset
from config import CONFIG, set_seed, device

set_seed(seed=42)
config = CONFIG()
warnings.filterwarnings("ignore", category=DeprecationWarning)

BACKBONE_TYPES = ['dinomcvit', 'dinomcvitsmall', 'dinomcvitbase', 'vitsmall', 'vitbase', 'vit']

DATASETS = {
    'uavid': {
        'module': uavid_dataset,
        'model_ckpt': config.best_models['UAVID']
    },
    'potsdam': {
        'module': potsdam_dataset,
        'model_ckpt': config.best_models['POTSDAM']
    },
    'loveda': {
        'module': loveda_dataset,
        'model_ckpt': config.best_models['LOVEDA']
    }
}

# Backbone mapping
BACKBONE_CONFIG = {
    'small': {
        'type': 'DinoMCViTSmall',
        'ckpt_key': 'vit_small'
    },
    'base': {
        'type': 'DinoMCViTBase',
        'ckpt_key': 'vit_base'
    }
}


def inference(image_path, output_path=None, image_size=224,
              backbone_type='DinoMCViT',
              backbone_size='small',  # 'small' or 'base'
              patch_size=16,
              dataset='uavid',  # 'uavid', 'potsdam', or 'loveda'
              visualize=True
              ):
    assert backbone_type.lower() in BACKBONE_TYPES
    assert dataset.lower() in DATASETS
    assert backbone_size.lower() in BACKBONE_CONFIG
    assert patch_size in [8, 16]

    dataset = dataset.lower()
    backbone_size = backbone_size.lower()

    dataset_info = DATASETS[dataset]
    dataset_module = dataset_info['module']
    seg_model_ckpt = dataset_info['model_ckpt']

    backbone_type = BACKBONE_CONFIG[backbone_size]['type']
    backbone_ckpt_key = BACKBONE_CONFIG[backbone_size]['ckpt_key']
    backbone_ckpt = config.dino_mc_checkpoint[backbone_ckpt_key][str(patch_size)]

    num_classes = len(dataset_module.CLASSES)
    class_names = dataset_module.CLASSES
    transform_fn = dataset_module.transform_test
    class_to_rgb = dataset_module.class_to_rgb
    class_palette = dataset_module.CLASS2PALETTE
    palette_class = dataset_module.PALETTE2CLASS

    model = load_trained_model(
        seg_model_ckpt=seg_model_ckpt, backbone_ckpt=backbone_ckpt,
        image_size=image_size,
        backbone_type=backbone_type,
        patch_size=patch_size,
        num_classes=num_classes,

    )

    img_tensor = preprocess_image(
        image_path=image_path, image_size=(image_size, image_size),
        transform_test=transform_fn
    )

    mask_tensor = predict_seg_mask(model, img_tensor)
    return mask_tensor.cpu(), class_to_rgb, palette_class, class_names


def load_trained_model(
        seg_model_ckpt, backbone_ckpt,
        image_size=224, backbone_type='DinoMCViTSmall',
        patch_size=16,
        num_classes=8
):
    print(f"[INFO] Loading model from {seg_model_ckpt}")
    model = UPerNetDinoMCViT(
        num_classes=num_classes,
        backbone_type=backbone_type,
        backbone_checkpoint=backbone_ckpt,
        img_size=image_size,
        patch_size=patch_size
    )
    checkpoint = torch.load(seg_model_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, transform_test, image_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    return transform_test(img, image_size=image_size).unsqueeze(0)  # shape: (1, 3, H, W)


@torch.no_grad()
def predict_seg_mask(model, image_tensor):
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)  # (1, C, H, W)
    output = F.interpolate(output, size=image_tensor.shape[2:], mode="bilinear", align_corners=False)
    mask = torch.argmax(output, dim=1)  # (1, H, W)
    return mask.squeeze(0).cpu()


def visualize_segmentation(
        image_path,
        pred_mask,
        class_to_rgb_fn=None,
        gt_mask_path=None,
        palette_to_class=None,
        class_names=None,
        final_size=(224, 224),
        show=True,
        figsize=(20, 8),
        font_size=16,
        legend_ncol=4,
):
    class_to_palette = {v: k for k, v in palette_to_class.items()}
    if final_size is None:
        w, h = pred_mask.shape[1], pred_mask.shape[0]
    else:
        w, h = final_size

    resize = A.Resize(height=h, width=w, interpolation=Image.BICUBIC)

    def resize_img(pil_img):
        return resize(image=np.array(pil_img))["image"]

    # Original image
    original_img = Image.open(image_path).convert("RGB")
    original_img = resize_img(original_img)

    # Prediction mask
    pred_rgb_mask = class_to_rgb_fn(pred_mask.numpy(), class_to_palette)
    pred_rgb_mask = resize_img(Image.fromarray(pred_rgb_mask))

    panels = [original_img]
    titles = ["Original Image"]

    if gt_mask_path and palette_to_class:
        gt_class_mask = load_gt_class_mask(gt_mask_path, palette_to_class)
        gt_rgb_mask = class_to_rgb_fn(gt_class_mask, class_to_palette)
        gt_rgb_mask = resize_img(Image.fromarray(gt_rgb_mask))
        panels.append(gt_rgb_mask)
        titles.append("Ground Truth Mask")

    panels.append(pred_rgb_mask)
    titles.append("Predicted Mask")

    # --- Plotting ---
    n_panels = len(panels)
    fig_height = 5
    fig_width = 5 * n_panels
    fig, axs = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))

    if n_panels == 1:
        axs = [axs]

    for ax, img, title in zip(axs, panels, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=font_size)
        ax.axis("off")

    # --- Legend ---
    if class_names:
        handles = [
            mpatches.Patch(
                color=np.array(class_to_palette[i]) / 255.0,
                label=class_names[i]
            ) for i in range(len(class_names))
        ]

        fig.subplots_adjust(bottom=0.25)
        fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.05),
            ncol=legend_ncol,
            fontsize=font_size - 2,
            frameon=False
        )

    # --- Save or show ---
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[INFO] Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def load_gt_class_mask(gt_path: str, palette_to_class: dict) -> np.ndarray:
    """
    Loads and converts ground truth segmentation mask to class indices.
    Supports:
        - 2D grayscale label masks (e.g., LOVEDA)
        - 3D RGB-encoded masks (e.g., UAVID, POTSDAM)

    Args:
        gt_path (str): Path to ground truth image.
        palette_to_class (dict): RGB tuple -> class index mapping (for RGB masks only).

    Returns:
        np.ndarray: Class mask (H, W)
    """
    gt_img = Image.open(gt_path)
    gt_np = np.array(gt_img)

    if len(gt_np.shape) == 2:  # 2D grayscale: direct class indices
        return gt_np.astype(np.uint8)

    elif len(gt_np.shape) == 3 and gt_np.shape[2] == 3:  # RGB mask
        h, w, _ = gt_np.shape
        flat_rgb = gt_np.reshape(-1, 3)
        class_mask_flat = np.zeros((flat_rgb.shape[0],), dtype=np.uint8)

        for i, rgb in enumerate(map(tuple, flat_rgb)):
            class_mask_flat[i] = palette_to_class.get(rgb, 0)

        return class_mask_flat.reshape(h, w)

    else:
        raise ValueError(f"Unsupported mask format: shape={gt_np.shape}")


def save_prediction(rgb_mask, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(rgb_mask).save(output_path)
    print(f"[INFO] Saved prediction mask to: {output_path}")


if __name__ == "__main__":
    image_size = config.image_size  # 224

    test_image_path = "tmp/images_to_test/seq1_000900.png"  # path to input image
    save_path = "tmp/predictions/seq1_000900_mask_pred.png"
    val_path = "tmp/images_to_test/seq1_000900_mask.png"

    # test_image_path = "tmp/images_to_test/3.png"  # path to input image
    # save_path = "tmp/predictions/3_mask_pred.png"
    # val_path = "tmp/images_to_test/3_mask.png"

    # test_image_path = "tmp/images_to_test/Image_24.tif"  # path to input image
    # save_path = "tmp/predictions/Image_24_mask_pred.tif"
    # val_path = "tmp/images_to_test/Label_24.tif"

    mask_tensor, class_to_rgb, palette_class, class_names = inference(
        image_path=test_image_path,
        output_path=save_path,
        image_size=image_size,
        backbone_type='DinoMCViT',
        backbone_size='small',  # 'small' or 'base'
        patch_size=8,
        dataset='uavid',  # 'uavid', 'potsdam', or 'loveda'
    )

    visualize_segmentation(
        image_path=test_image_path,
        pred_mask=mask_tensor,
        gt_mask_path=val_path,
        class_to_rgb_fn=class_to_rgb,
        palette_to_class=palette_class,
        final_size=(512, 512),
        class_names=class_names,
    )
