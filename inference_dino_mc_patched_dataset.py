import os, numpy as np, torch
from tqdm import tqdm

from models.networks import UPerNetDinoMC
from data_process.uavid_dataset import PatchedUAVIDDataset
from config import CONFIG, set_seed, device

set_seed(42)
config = CONFIG()


def load_dino_mc_seg_model(seg_ckpt, backbone_ckpt, n_classes, img_sz, patch_sz):
    model = UPerNetDinoMC(
        num_classes=n_classes,
        backbone_type='vit_small',
        pretrained_ckpt=backbone_ckpt,
        img_size=img_sz,
        patch_size=patch_sz,
        feature_stack='pyramid',
        use_neck=False
    )
    state = torch.load(seg_ckpt, map_location=device)
    key = "model_state_dict" if "model_state_dict" in state else "state_dict"
    model.load_state_dict(state[key])
    return model.to(device).eval()


@torch.no_grad()
def export_dino_mc_split_logits(split: str, batch: int, seg_ckpt, out_dir: str = None, data_root=None):
    img_sz = config.image_size

    inf_dataset = PatchedUAVIDDataset(
        split, img_size=config.image_size, data_root=data_root
    )
    inf_dataloader = torch.utils.data.DataLoader(
        inf_dataset, batch_size=batch,
        shuffle=False, num_workers=4,
        # pin_memory=True
    )

    bck = os.path.join(config.model_weights_path,
                       config.dino_mc["vit_small"][str(config.patch_size)])
    model = load_dino_mc_seg_model(seg_ckpt, bck, inf_dataset.num_classes,
                                   config.image_size, config.patch_size)

    os.makedirs(out_dir, exist_ok=True)

    for imgs, names in tqdm(inf_dataloader, desc=f"[{split}]"):
        imgs = imgs.to(device, non_blocking=True)

        # mmseg needs img_meta
        meta = [{
            "img_shape": (img_sz, img_sz, 3),
            "ori_shape": (img_sz, img_sz, 3),
            "pad_shape": (img_sz, img_sz, 3),
            "scale_factor": 1.0,
        } for _ in range(imgs.size(0))]

        logits = model.encode_decode(imgs, meta).cpu()  # (B,C,H,W)

        for logit, stem in zip(logits, names):
            np.save(os.path.join(out_dir, f"{stem}.npy"),
                    logit.numpy().astype(np.float32))

    print(f"✓  {split}: {len(inf_dataset)} patch-logits saved → {out_dir}")


if __name__ == "__main__":
    patche_count = 4
    batch_size = config.batch_size
    seg_ckpt = config.best_models['UAVID_patched']['dino_mc'][str(patche_count)]

    for split in ['train', 'val']:
        data_root = config.UAVID_patched['4']['no_overlap'][split]
        save_dir = config.UAVID_patch_inf['dino_mc']['4']['no_overlap'][split]
        os.makedirs(save_dir, exist_ok=True)
        export_dino_mc_split_logits(
            split=split, batch=batch_size,
            seg_ckpt=seg_ckpt, out_dir=save_dir,
            data_root=data_root
        )
