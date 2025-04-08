import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_process.uavid_dataset import UAVIDDataset
from data_process.potsdam_dataset import POTSDAMDataset
from data_process.loveda_dataset import LOVEDADataset
from utils.metric_tool import (
    print_class_metrics_table, calculate_per_class_metrics,
    write_epoch_csv
)
from utils.loss import DiceCrossEntropyLoss
from models.networks import UPerNetDinoMCViT

from config import CONFIG, set_seed

set_seed(seed=42)
config = CONFIG()
wandb.login(key=config.wandb_api_key)


def train_model(
        model, train_loader, val_loader,
        epochs=100, device='cuda',
        save_dir='./checkpoints',
        experiment_name='11'
):
    os.makedirs(save_dir, exist_ok=True)
    csv_file_name = f'epoch_results_{model.backbone_type.lower()}_{experiment_name}.csv'
    csv_path = os.path.join(save_dir, csv_file_name)

    lr = 3e-4
    weight_decay = 1e-4
    eta_min = 1e-6
    interpolation_mode = 'bilinear'

    wandb.init(
        project="DINO MC Segmentation",
        name=f"experiment_{experiment_name}",
        config={
            "learning_rate": lr,
            "architecture": "ViT",
            # "dataset": "LOVEDA",
            "dataset": "POTSDAM",
            # "dataset": "UAVID",
            "interpolation_mode": interpolation_mode,
            "weight_decay": weight_decay,
            "eta_min": eta_min,
            "epochs": epochs,
            "patch_size": patch_size
        })

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)

    # TODO: loss_function.dice_bce_loss()
    # criterion = DiceCrossEntropyLoss(ignore_index=0)
    criterion = DiceCrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss()

    best_miou = 0
    best_epoch = 0
    class_names = train_loader.dataset.class_names

    epoch_data = {}  # { epoch_number : {}) }
    for epoch in range(1, epochs + 1):
        ########## Train Phase ##########
        model.train()
        total_loss = 0
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} - Training")):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            # TODO train using bicubic
            # outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bicubic', align_corners=False)

            outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        ########## Validation Phase ##########
        model.eval()
        total_val_loss = 0
        preds_list, masks_list = [], []
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} - Validation")):
                images = images.to(device)
                # seg_logits = model.inference(images, rescale=True)
                seg_logits = model(images)
                seg_logits = nn.functional.interpolate(seg_logits, size=masks.shape[-2:], mode='bilinear',
                                                       align_corners=False)

                seg_logits = seg_logits.to('cpu')

                val_loss = criterion(seg_logits, masks)
                total_val_loss += val_loss.item()

                preds = seg_logits.argmax(dim=1)

                preds_list.append(preds)
                masks_list.append(masks)
        avg_val_loss = total_val_loss / len(val_loader)

        preds_all = torch.cat(preds_list)
        masks_all = torch.cat(masks_list)

        metrics_dict = calculate_per_class_metrics(preds_all, masks_all, class_names=class_names)
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
        epoch_data[epoch] = metrics_dict

        # Now update (write) the CSV after each epoch
        write_epoch_csv(epoch_data, csv_path)

    print(f"\nTraining complete. Best IoU={best_miou:.4f} at epoch={best_epoch}.")
    wandb.finish()

if __name__ == "__main__":
    from config import CONFIG

    config = CONFIG()
    backbone_type = 'vit_small'
    patch_size = 16
    backbone_ckpt = config.dino_mc_checkpoint[backbone_type][str(patch_size)]
    batch_size = config.batch_size
    image_size = 224

    # UAVID training
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

    # # Potsdam training
    # train_data_root = config.POTSDAM['train']
    # val_data_root = config.POTSDAM['val']
    #
    # train_dataset = POTSDAMDataset(
    #     root_dir=train_data_root, mode='train', image_size=image_size
    # )
    # val_dataset = POTSDAMDataset(
    #     root_dir=val_data_root, mode='val', image_size=image_size
    # )

    # # LOVEDA training
    # train_data_root = config.LOVEDA['train']
    # val_data_root = config.LOVEDA['val']
    #
    # train_dataset = LOVEDADataset(
    #     root_dir=train_data_root, mode='train', image_size=image_size
    # )
    # val_dataset = LOVEDADataset(
    #     root_dir=val_data_root, mode='val', image_size=image_size
    # )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = train_dataset.num_classes
    model = UPerNetDinoMCViT(
        num_classes=num_classes,
        backbone_type='DinoMCViTSmall',
        backbone_checkpoint=backbone_ckpt,
        img_size=image_size
    )

    train_model(model, train_loader, val_loader, epochs=100, device='cuda',
                # save_dir='./checkpoints', experiment_name='loveda_2')
                save_dir='./checkpoints', experiment_name='uavid_2_with_transformation')
                # save_dir='./checkpoints', experiment_name='potsdam_2')