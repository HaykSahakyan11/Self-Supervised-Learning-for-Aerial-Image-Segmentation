import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from models.networks import UPerNetDinoMCViT
from data_process import uavid_dataset, potsdam_dataset, loveda_dataset
from config import CONFIG, set_seed, device

set_seed(seed=42)
config = CONFIG()

# DATASET_NAMES = ['uavid', 'potsdam', 'loveda']
BACKBONE_TYPES = ['dinomcvit', 'dinomcvitsmall', 'dinomcvitbase', 'vitsmall', 'vitbase', 'vit']
# BACKBONE_SIZES = ['small', 'base']

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
    # TODO for testing
    # seg_model_ckpt = "checkpoints/best_checkpoint_dinomcvitsmall_uavid_2_with_transformation.pth"
    # seg_model_ckpt = "checkpoints/best_checkpoint_dinomcvitsmall_potsdam_2.pth"
    # seg_model_ckpt = "checkpoints/best_checkpoint_dinomcvitsmall_loveda_2.pth"

    backbone_type = BACKBONE_CONFIG[backbone_size]['type']
    backbone_ckpt_key = BACKBONE_CONFIG[backbone_size]['ckpt_key']
    backbone_ckpt = config.dino_mc_checkpoint[backbone_ckpt_key][str(patch_size)]

    num_classes = len(dataset_module.CLASSES)
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
    return mask_tensor.cpu(), class_to_rgb, palette_class


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


def visualize_result(image_path, mask_tensor, rgb_mask=None):
    original_img = Image.open(image_path).convert("RGB")
    original_img = original_img.resize(mask_tensor.shape[::-1], resample=Image.BICUBIC)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(rgb_mask)
    axs[1].set_title("Predicted Segmentation")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()


def visualize_segmentation(
        image_path,
        pred_mask,
        class_to_rgb_fn=None,
        gt_mask_path=None,
        palette_to_class=None,
        figsize=(18, 6)
):
    class_to_palette = {v: k for k, v in palette_to_class.items()}
    pred_rgb_mask = class_to_rgb(pred_mask.numpy(), class_to_palette)

    original_img = Image.open(image_path).convert("RGB")
    original_img = original_img.resize(pred_rgb_mask.shape[:2][::-1], resample=Image.BICUBIC)

    panels = [original_img]
    titles = ["Original Image"]

    # TODO: Add a check for gt_mask_path
    if gt_mask_path and palette_to_class:
        gt_rgb = Image.open(gt_mask_path).convert("RGB")
        gt_np = np.array(gt_rgb)
        gt_class_mask = np.zeros((gt_np.shape[0], gt_np.shape[1]), dtype=np.uint8)

        for rgb, class_idx in palette_to_class.items():  # Use PALETTE2CLASS
            match = np.all(gt_np == rgb, axis=-1)
            gt_class_mask[match] = class_idx

        gt_rgb_mask = class_to_rgb_fn(gt_class_mask, class_to_palette)  # use CLASS2PALETTE for RGB coloring
        panels.append(gt_rgb_mask)
        titles.append("Ground Truth Mask")

    panels.append(pred_rgb_mask)
    titles.append("Predicted Mask")

    fig, axs = plt.subplots(1, len(panels), figsize=figsize)
    for ax, img, title in zip(axs, panels, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def save_prediction(rgb_mask, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(rgb_mask).save(output_path)
    print(f"[INFO] Saved prediction mask to: {output_path}")


if __name__ == "__main__":
    image_size = config.image_size  # 224

    # test_image_path = "tmp/images_to_test/seq1_000900.png"  # path to input image
    # save_path = "tmp/predictions/seq1_000900_mask_pred.png"
    # val_path = "tmp/images_to_test/seq1_000900_mask.png"

    test_image_path = "tmp/images_to_test/3.png"  # path to input image
    save_path = "tmp/predictions/3_mask_pred.png"
    val_path = "tmp/images_to_test/3_mask.png"

    # test_image_path = "tmp/images_to_test/Image_24.tif"  # path to input image
    # save_path = "tmp/predictions/Image_24_mask_pred.tif"
    # save_path = "tmp/images_to_test/Label_24.tif"

    mask_tensor, class_to_rgb, palette_class = inference(
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
        palette_to_class=palette_class
    )
