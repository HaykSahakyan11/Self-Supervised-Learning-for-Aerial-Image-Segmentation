# train_dino_mc_seg.py

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config import CONFIG, set_seed
from data_process.uavid_dataset import UAVIDDataset
from utils.loss import DiceCrossEntropyLoss
from utils.metric_tool import calculate_per_class_metrics, print_class_metrics_table, write_epoch_csv
from models.networks import DinoMCBackbone, UPerNetDinoMC

set_seed(seed=42)
config = CONFIG()
wandb.login(key=config.wandb_api_key)


def freeze_module(m):
    for p in m.parameters():
        p.requires_grad = False


def train_model(
        model, train_loader, val_loader,
        epochs=100, device='cuda',
        save_dir='./checkpoints',
        experiment_name='11',
        train_backbone=True,
        train_decoder=True
):
    os.makedirs(save_dir, exist_ok=True)
    csv_file_name = f'epoch_results_{model.backbone_type.lower()}_{experiment_name}.csv'
    csv_path = os.path.join(save_dir, csv_file_name)

    lr = config.train_configs['learning_rate']
    weight_decay = config.train_configs['weight_decay']
    eta_min = config.train_configs['eta_min']
    interpolation_mode = 'bilinear'

    wandb.init(
        project="DINO MC Segmentation_v2",
        name=f"experiment_{experiment_name}",
        config={
            "learning_rate": lr,
            "architecture": "ViT",
            # "dataset": "LOVEDA",
            # "dataset": "POTSDAM",
            "dataset": "UAVID",
            "decode": "upernet_uperhead_patched_features_pyramid",
            "interpolation_mode": interpolation_mode,
            "weight_decay": weight_decay,
            "eta_min": eta_min,
            "epochs": epochs,
            "patch_size": 8
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer & scheduler
    if not train_backbone:
        freeze_module(model.backbone)

    if not train_decoder:
        freeze_module(model.decode_head)
        freeze_module(model.auxiliary_head)
        if model.use_neck:
            # TODO not working model.neck
            freeze_module(model.neck)
    if not train_backbone and not train_decoder:
        raise ValueError("Both backbone and decoder are frozen. Nothing to train.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=eta_min
    )

    # loss
    criterion = DiceCrossEntropyLoss()

    best_miou = 0
    best_epoch = 0
    class_names = train_loader.dataset.class_names

    history = {}
    for epoch in range(1, epochs + 1):


        ########## Train Phase ##########
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # build meta
            img_meta = [{
                'img_shape': (images.size(2), images.size(3), 3),
                'ori_shape': (images.size(2), images.size(3), 3),
                'pad_shape': (images.size(2), images.size(3), 3),
                'scale_factor': 1.0,
            } for _ in range(images.size(0))]

            # forward through encoder+decoder, gives full‑res logits
            logits = model.encode_decode(images, img_meta)

            # logits = model(imgs)  # (B, C, H, W)
            # logits = F.interpolate(
            #     logits, size=masks.shape[-2:],
            #     mode="bilinear", align_corners=False
            # )
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        ########## Validation Phase ##########
        model.eval()
        total_val_loss = 0.0
        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                images = images.to(device)
                # images, masks = images.to(device), masks.to(device)

                img_meta = [{
                    'img_shape': (images.size(2), images.size(3), 3),
                    'ori_shape': (images.size(2), images.size(3), 3),
                    'pad_shape': (images.size(2), images.size(3), 3),
                    'scale_factor': 1.0,
                } for _ in range(images.size(0))]
                logits = model.encode_decode(images, img_meta)
                logits = logits.to('cpu')

                # logits = model(images)
                # logits = F.interpolate(
                #     logits.cpu(), size=masks.shape[-2:],
                #     mode="bilinear", align_corners=False
                # )
                val_loss = criterion(logits, masks)
                total_val_loss += val_loss.item()

                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_masks.append(masks)

        avg_val_loss = total_val_loss / len(val_loader)
        preds_cat = torch.cat(all_preds)
        masks_cat = torch.cat(all_masks)

        metrics_dict = calculate_per_class_metrics(
            preds_cat, masks_cat, class_names=class_names
        )
        acc, mean_iou, mean_f1 = metrics_dict['accuracy'], metrics_dict['mean_iou'], metrics_dict['mean_f1']

        print_class_metrics_table(metrics_dict)
        print(f"\nEpoch [{epoch}/{epochs}] - Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {acc:.4f}, Mean IoU: {mean_iou:.4f}, Mean F1: {mean_f1:.4f}")

        ######### wandb Logging ##########
        log_dict = {
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/accuracy": acc,
            "val/mean_iou": mean_iou,
            "val/mean_f1": mean_f1,
        }
        # Log per-class metrics
        for idx, cname in enumerate(class_names):
            log_dict[f"val/{cname}_iou"] = metrics_dict['per_class_iou'][idx]
            log_dict[f"val/{cname}_f1"] = metrics_dict['per_class_f1'][idx]
            log_dict[f"val/{cname}_acc"] = metrics_dict['per_class_precision'][idx]

        wandb.log(log_dict)

        if mean_iou > best_miou:
            best_miou = mean_iou
            best_epoch = epoch
            save_path = os.path.join(save_dir, f'best_checkpoint_{model.backbone_type.lower()}_{experiment_name}.pth')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou
            }, save_path)
            print(f"** Best model updated at epoch {best_epoch}, IoU = {best_miou:.4f}. Saved to {save_path}\n")

        scheduler.step()

        # Collect data for CSV
        history[epoch] = metrics_dict

        # Now update (write) the CSV after each epoch
        write_epoch_csv(history, csv_path)

    print(f"\nTraining finished. Best IoU={best_miou:.4f} at epoch={best_epoch}.")
    wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    cfg = CONFIG()

    # Data
    train_ds = UAVIDDataset(
        data_root=cfg.UAVID_patched["4"]["no_overlap"]["train"],
        # data_root=cfg.UAVID_patched["9"]["no_overlap"]["train"],
        # data_root=cfg.UAVID_patched['224_224']["train"],
        # data_root=cfg.UAVID_patched['360_384']["train"],
        img_dir="images", mask_dir="labels",
        image_size=cfg.image_size
    )
    val_ds = UAVIDDataset(
        data_root=cfg.UAVID_patched["4"]["no_overlap"]["val"],
        # data_root=cfg.UAVID_patched["9"]["no_overlap"]["val"],
        # data_root=cfg.UAVID_patched['224_224']["val"],
        # data_root=cfg.UAVID_patched['360_384']["val"],
        img_dir="images", mask_dir="labels",
        image_size=cfg.image_size
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # Model
    patch_size = cfg.patch_size
    ckpt_file = cfg.dino_mc["vit_small"][str(cfg.patch_size)]
    pretrained_ckpt = os.path.join(cfg.model_weights_path, ckpt_file)

    model = UPerNetDinoMC(
        num_classes=train_ds.num_classes,
        backbone_type='vit_small',
        pretrained_ckpt=pretrained_ckpt,
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        feature_stack='pyramid',
        use_neck=False
    )
    model.init_weights()

    # Train
    train_model(model, train_loader, val_loader, epochs=100, device="cuda",
                # save_dir='./checkpoints', experiment_name='uavid_upernet_backbone_ft_decod_tr_patched_224_224_no_overlap',
                # save_dir='./checkpoints', experiment_name='uavid_upernet_backbone_ft_decod_tr_patched_360_384_no_overlap',
                save_dir='./checkpoints', experiment_name='uavid_upernet_backbone_no_ft_decod_tr_patched_4_no_overlap_aa',
                train_backbone=False, train_decoder=True)
