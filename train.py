import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import datetime

from dataset import BTXRD_Dataset
from models.networks.prompt_unet_2D import PGA_UNet

# =========================================================
# CẤU HÌNH – chỉ đổi ở đây khi chuyển thí nghiệm
# =========================================================
EXPERIMENT         = 'B'     # 'A' → zoom_out only | 'B' → mixed_7_3
USE_ENCODER_PROMPT = True    # True để bật PromptSpatialGate ở encoder
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS     = 100
LR         = 1e-4
IMG_SIZE   = 512
EARLY_STOP = 15

# =========================================================
# METRICS
# =========================================================
def dice_loss(pred, target, smooth=1e-5):
    pred_soft    = torch.sigmoid(pred)
    intersection = (pred_soft * target).sum(dim=(1, 2, 3))
    union        = pred_soft.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - ((2. * intersection + smooth) / (union + smooth))).mean()


def batch_metrics_sum(pred, target, smooth=1e-5):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))
    dice      = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall    = tp / (tp + fn + smooth)
    return dice.sum().item(), iou.sum().item(), precision.sum().item(), recall.sum().item()


def calculate_cbl(pred, target, smooth=1e-6):
    """
    CBL – Center-Based Localization score [0, 1].
    Đo tâm predicted mask có gần tâm GT mask không.
    Normalize theo đường chéo GT bbox để scale-invariant.
    Trả về (cbl_sum, valid_count).
    """
    B, _, H, W = pred.shape
    pred_bin = (torch.sigmoid(pred) > 0.5).float()

    ys = torch.arange(H, device=pred.device, dtype=torch.float32)
    xs = torch.arange(W, device=pred.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

    cbl_sum, valid_count = 0.0, 0

    for b in range(B):
        gt_m   = target[b, 0]
        pred_m = pred_bin[b, 0]
        gt_area = gt_m.sum()

        if gt_area < smooth:
            continue  # GT rỗng → bỏ qua

        # Tâm GT
        cx_gt = (grid_x * gt_m).sum() / (gt_area + smooth)
        cy_gt = (grid_y * gt_m).sum() / (gt_area + smooth)

        # Đường chéo GT bbox
        nz    = gt_m.nonzero()
        gt_diag = torch.sqrt(
            ((nz[:, 0].max() - nz[:, 0].min()).float()) ** 2 +
            ((nz[:, 1].max() - nz[:, 1].min()).float()) ** 2
        ) + smooth

        pred_area = pred_m.sum()
        if pred_area < smooth:
            valid_count += 1  # CBL = 0 cho sample này
            continue

        # Tâm predicted mask
        cx_p = (grid_x * pred_m).sum() / (pred_area + smooth)
        cy_p = (grid_y * pred_m).sum() / (pred_area + smooth)

        d   = torch.sqrt((cx_p - cx_gt) ** 2 + (cy_p - cy_gt) ** 2)
        cbl = torch.clamp(1.0 - d / gt_diag, min=0.0)
        cbl_sum += cbl.item()
        valid_count += 1

    return cbl_sum, valid_count


# =========================================================
# LOGGER
# =========================================================
def setup_logger(exp_name):
    os.makedirs("logs", exist_ok=True)
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[
            logging.FileHandler(f"logs/train_exp{exp_name}_{t}.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


# =========================================================
# MAIN
# =========================================================
def main():
    logger = setup_logger(EXPERIMENT)
    logger.info("=" * 90)
    logger.info(f"Thí nghiệm {EXPERIMENT} | Device: {DEVICE} | EncoderPrompt: {USE_ENCODER_PROMPT}")
    logger.info("=" * 90)

    # ── Dataset ──────────────────────────────────────────────────────
    train_mode = 'zoom_out' if EXPERIMENT == 'A' else 'mixed_7_3'

    train_ds = BTXRD_Dataset(
        image_dir="dataset_BTXRD/train/images",
        json_dir="dataset_BTXRD/train/annotations",
        img_size=IMG_SIZE, is_train=True,
        prompt_mode=train_mode
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)

    # Val loaders: Exp A → 1 loader; Exp B → 3 loaders
    val_cfg = {'zoom_out': 'zoom_out'} if EXPERIMENT == 'A' else {
        'zoom_out':  'zoom_out',
        'shift':     'shift',
        'mixed_7_3': 'mixed_7_3',
    }
    val_loaders = {}
    for name, mode in val_cfg.items():
        ds = BTXRD_Dataset(
            image_dir="dataset_BTXRD/val/images",
            json_dir="dataset_BTXRD/val/annotations",
            img_size=IMG_SIZE, is_train=False,
            prompt_mode=mode
        )
        val_loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=2, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────
    model = PGA_UNet(in_channels=1, n_classes=1,
                     use_encoder_prompt=USE_ENCODER_PROMPT).to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer     = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_dice   = 0.0
    patience_counter = 0
    ckpt_prefix     = f"checkpoints/pga_unet_exp{EXPERIMENT}"

    for epoch in range(EPOCHS):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, masks, prompts in loop:
            images, masks, prompts = (images.to(DEVICE), masks.to(DEVICE), prompts.to(DEVICE))
            preds = model(images, prompts)
            loss  = criterion_bce(preds, masks) + dice_loss(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss_avg = train_loss / len(train_loader)

        # ── Validate (tất cả loaders) ─────────────────────────────
        model.eval()
        val_results = {}
        with torch.no_grad():
            for vname, vloader in val_loaders.items():
                s_dice, s_iou, s_pre, s_rec = 0, 0, 0, 0
                s_cbl, n_cbl, n_total = 0, 0, 0
                for vi, vm, vp in vloader:
                    vi, vm, vp = vi.to(DEVICE), vm.to(DEVICE), vp.to(DEVICE)
                    vout = model(vi, vp)
                    d, i, p, r = batch_metrics_sum(vout, vm)
                    s_dice += d; s_iou += i; s_pre += p; s_rec += r
                    cb, ncb = calculate_cbl(vout, vm)
                    s_cbl += cb; n_cbl += ncb
                    n_total += vi.size(0)
                val_results[vname] = {
                    'dice': s_dice / n_total,
                    'iou':  s_iou  / n_total,
                    'pre':  s_pre  / n_total,
                    'rec':  s_rec  / n_total,
                    'cbl':  s_cbl  / n_cbl if n_cbl > 0 else 0.0,
                }

        # Dùng zoom_out dice làm tiêu chí chính
        primary_dice = val_results['zoom_out']['dice']
        scheduler.step(primary_dice)

        # Log
        log_str = f"Epoch {epoch+1:3d} | T_Loss: {train_loss_avg:.4f}"
        for vname, vr in val_results.items():
            tag = vname.upper()[:7]
            log_str += (f" | [{tag}] Dice:{vr['dice']:.4f} IoU:{vr['iou']:.4f}"
                        f" CBL:{vr['cbl']:.4f}")
        log_str += f" | LR:{optimizer.param_groups[0]['lr']:.1e}"

        torch.save(model.state_dict(), f"{ckpt_prefix}_last.pth")

        if primary_dice > best_val_dice:
            best_val_dice = primary_dice
            torch.save(model.state_dict(), f"{ckpt_prefix}_best.pth")
            log_str = "🥇 [BEST] " + log_str
            patience_counter = 0
        else:
            patience_counter += 1

        logger.info(log_str)

        if patience_counter >= EARLY_STOP:
            logger.info(f"Early stopping ở epoch {epoch+1}.")
            break

    logger.info(f"\nBest Dice (zoom_out): {best_val_dice:.4f}")
    logger.info(f"Checkpoint: {ckpt_prefix}_best.pth")


if __name__ == "__main__":
    main()