import os, numpy as np, torch
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image

from models.networks import UPerNetDinoDeiT
from data_process.uavid_dataset import PatchedUAVIDDataset
from config import CONFIG, set_seed, device

set_seed(42)
config = CONFIG()


def load_dino_deit_seg_model(seg_ckpt, n_classes, img_sz, patch_sz, backbone_ckpt=None):
    model = UPerNetDinoDeiT(
        num_classes=n_classes,
        backbone_type='DinoDeiTSmall',
        img_size=img_sz,
        patch_size=patch_sz,
        feature_stack='pyramid',
        use_neck=False,
        pretrained_ckpt=None
    )
    state = torch.load(seg_ckpt, map_location=device)
    key = "model_state_dict" if "model_state_dict" in state else "state_dict"
    model.load_state_dict(state[key])
    return model.to(device).eval()


@torch.no_grad()
def export_dino_deit_split_logits(split: str, batch: int, seg_ckpt, out_dir: str = None, data_root: str = None):
    img_sz = config.image_size
    inf_dataset = PatchedUAVIDDataset(
        split, img_size=config.image_size, data_root=data_root
    )
    inf_dataloader = torch.utils.data.DataLoader(
        inf_dataset, batch_size=batch,
        shuffle=False, num_workers=4
    )

    bck = os.path.join(config.model_weights_path,
                       config.dino_deit['vit_small'][str(config.patch_size)])
    model = load_dino_deit_seg_model(seg_ckpt, inf_dataset.num_classes,
                                   config.image_size, config.patch_size, backbone_ckpt=bck)
    os.makedirs(out_dir, exist_ok=True)

    for imgs, names in tqdm(inf_dataloader, desc=f"{split}"):
        imgs = imgs.to(device, non_blocking=True)
        logits = F.interpolate(model(imgs), size=imgs.shape[-2:],
                               mode='bilinear', align_corners=False).cpu()

        for logit, name in zip(logits, names):
            np.save(os.path.join(out_dir, f"{name}.npy"),
                    logit.numpy().astype(np.float32))

    print(f"✓ {split}: saved {len(inf_dataset)} patch‑logits to {out_dir}")


if __name__ == "__main__":
    patche_count = 4
    batch_size = config.batch_size
    # seg_ckpt = config.best_model_weights['upernet_dinodeit']['vit_small']['uavid']['patch_4']
    seg_ckpt = config.best_models['UAVID_patched']['dino_deit'][str(patche_count)]

    for split in ['train', 'val']:
        data_root = config.UAVID_patched['4'][split]
        save_dir = config.UAVID_patch_inf['dino_deit']['4'][split]
        os.makedirs(save_dir, exist_ok=True)
        export_dino_deit_split_logits(
            split=split, batch=batch_size,
            seg_ckpt=seg_ckpt, out_dir=save_dir,
            data_root=data_root
        )
