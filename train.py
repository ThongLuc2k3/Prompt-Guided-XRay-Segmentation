import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import datetime

from dataset import BTXRD_Dataset
from models.networks.prompt_unet_2D import PGA_UNet  # Import đúng tên Class

# Hàm Loss và Metrics giữ nguyên như của bạn vì logic tính tổng chia sau là cực chuẩn
def dice_loss(pred, target, smooth=1e-5):
    pred_soft = torch.sigmoid(pred)
    intersection = (pred_soft * target).sum(dim=(1,2,3))
    union = pred_soft.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def calculate_batch_metrics_sum(pred, target, smooth=1e-5):
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    tp = (pred_binary * target).sum(dim=(1,2,3))
    fp = (pred_binary * (1 - target)).sum(dim=(1,2,3))
    fn = ((1 - pred_binary) * target).sum(dim=(1,2,3))
    
    dice_score = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
    iou_score = (tp + smooth) / (tp + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    return dice_score.sum().item(), iou_score.sum().item(), precision.sum().item(), recall.sum().item()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 100 # Tăng lên vì đã có Early Stopping
LR = 1e-4
IMG_SIZE = 512

def setup_logger():
    os.makedirs("logs", exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_baseline_{time_str}.log"
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
    return logging.getLogger()

def main():
    logger = setup_logger()
    logger.info("="*95)
    logger.info(f"🚀 KHỞI ĐỘNG HUẤN LUYỆN PGA-UNET - THIẾT BỊ: {DEVICE}")
    logger.info("="*95)

    train_dataset = BTXRD_Dataset(
        image_dir="dataset_BTXRD/train/images", 
        json_dir="dataset_BTXRD/train/annotations", 
        img_size=IMG_SIZE, is_train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = BTXRD_Dataset(
        image_dir="dataset_BTXRD/val/images", 
        json_dir="dataset_BTXRD/val/annotations", 
        img_size=IMG_SIZE, is_train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PGA_UNet(in_channels=1, n_classes=1).to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Thêm Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_dice = 0.0
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 15

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, masks, prompts in loop:
            images, masks, prompts = images.to(DEVICE), masks.to(DEVICE), prompts.to(DEVICE)
            
            predictions = model(images, prompts)
            loss = criterion_bce(predictions, masks) + dice_loss(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping bảo vệ Attention Gate
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss_avg = train_loss / len(train_loader)

        model.eval()
        val_loss, sum_dice, sum_iou, sum_pre, sum_rec = 0, 0, 0, 0, 0
        total_val_samples = 0
        
        with torch.no_grad():
            for val_images, val_masks, val_prompts in val_loader:
                val_images, val_masks, val_prompts = val_images.to(DEVICE), val_masks.to(DEVICE), val_prompts.to(DEVICE)
                val_preds = model(val_images, val_prompts)
                            
                v_loss = criterion_bce(val_preds, val_masks) + dice_loss(val_preds, val_masks)
                val_loss += v_loss.item()
                
                d, i, p, r = calculate_batch_metrics_sum(val_preds, val_masks)
                sum_dice += d; sum_iou += i; sum_pre += p; sum_rec += r
                total_val_samples += val_images.size(0)
                
        val_loss_avg = val_loss / len(val_loader)
        val_dice_avg = sum_dice / total_val_samples
        val_iou_avg = sum_iou / total_val_samples
        val_pre_avg = sum_pre / total_val_samples
        val_rec_avg = sum_rec / total_val_samples

        # Điều chỉnh Learning Rate
        scheduler.step(val_dice_avg)

        torch.save(model.state_dict(), "checkpoints/pga_unet_last.pth")
        log_str = (f"Epoch {epoch+1} | T_Loss: {train_loss_avg:.4f} | V_Loss: {val_loss_avg:.4f} | "
                   f"Dice: {val_dice_avg:.4f} | IoU: {val_iou_avg:.4f} | LR: {optimizer.param_groups[0]['lr']}")
        
        if val_dice_avg > best_val_dice:
            best_val_dice = val_dice_avg
            torch.save(model.state_dict(), "checkpoints/pga_unet_best.pth")
            log_str = "🥇 [BEST] " + log_str
            patience_counter = 0 # Reset đếm ngược
        else:
            patience_counter += 1
            
        logger.info(log_str)
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"🛑 Kích hoạt Early Stopping ở epoch {epoch+1}. Validation Dice không tăng trong {EARLY_STOPPING_PATIENCE} epochs.")
            break

if __name__ == "__main__":
    main()