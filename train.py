import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging
import datetime

# Import Dataset và Model của bạn
from dataset import BTXRD_Dataset
from models.networks.unet_2D import unet_2D

# ==========================================
# 1. HỆ THỐNG HÀM MẤT MÁT (Chuẩn hóa công thức)
# ==========================================
def dice_loss(pred, target, smooth=1e-5):
    pred_soft = torch.sigmoid(pred)
    # Tính theo từng ảnh trong batch (dim=(1,2,3) vì tensor [B, C, H, W])
    intersection = (pred_soft * target).sum(dim=(1,2,3))
    union = pred_soft.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# ==========================================
# 2. HỆ THỐNG ĐỘ ĐO THỰC TẾ (Khắc phục Bias Batch)
# ==========================================
def calculate_batch_metrics_sum(pred, target, smooth=1e-5):
    """Tính toán và trả về TỔNG ĐIỂM của toàn bộ ảnh trong batch, không lấy trung bình"""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    
    # Tính True Positive, False Positive, False Negative theo từng ảnh
    tp = (pred_binary * target).sum(dim=(1,2,3))
    fp = (pred_binary * (1 - target)).sum(dim=(1,2,3))
    fn = ((1 - pred_binary) * target).sum(dim=(1,2,3))
    
    # Tối ưu IoU theo công thức tp + fp + fn
    dice_score = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
    iou_score = (tp + smooth) / (tp + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    
    # Trả về tổng điểm của batch này
    return dice_score.sum().item(), iou_score.sum().item(), precision.sum().item(), recall.sum().item()

# ==========================================
# 3. CẤU HÌNH & HỆ THỐNG GHI LOG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
IMG_SIZE = 256

def setup_logger():
    os.makedirs("logs", exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_baseline_{time_str}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def main():
    logger = setup_logger()
    logger.info("="*95)
    logger.info(f"🚀 KHỞI ĐỘNG HUẤN LUYỆN BASELINE - THIẾT BỊ: {DEVICE}")
    logger.info(f"Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
    logger.info("="*95)

    # Khởi tạo Dataloader
    train_dataset = BTXRD_Dataset(image_dir="dataset_BTXRD/train/images", mask_dir="dataset_BTXRD/train/masks", img_size=IMG_SIZE, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = BTXRD_Dataset(image_dir="dataset_BTXRD/val/images", mask_dir="dataset_BTXRD/val/masks", img_size=IMG_SIZE, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Khởi tạo Model, Loss, Optimizer
    model = unet_2D(in_channels=1, n_classes=1).to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_dice = 0.0

    for epoch in range(EPOCHS):
        # --- PHA 1: TRAIN ---
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            predictions = model(images)
            loss = criterion_bce(predictions, masks) + dice_loss(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss_avg = train_loss / len(train_loader)

        # --- PHA 2: VALIDATION CHUẨN THỐNG KÊ ---
        model.eval()
        val_loss = 0
        sum_dice, sum_iou, sum_pre, sum_rec = 0, 0, 0, 0
        total_val_samples = 0 # Đếm tổng số ảnh thực tế
        
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(DEVICE), val_masks.to(DEVICE)
                batch_size_current = val_images.size(0)

                val_preds = model(val_images)
                
                v_loss = criterion_bce(val_preds, val_masks) + dice_loss(val_preds, val_masks)
                val_loss += v_loss.item()
                
                # Lấy TỔNG điểm của batch
                d, i, p, r = calculate_batch_metrics_sum(val_preds, val_masks)
                sum_dice += d
                sum_iou += i
                sum_pre += p
                sum_rec += r
                
                total_val_samples += batch_size_current
                
        # Tính điểm trung bình chuẩn xác bằng cách chia cho TỔNG SỐ ẢNH
        val_loss_avg = val_loss / len(val_loader) # Loss thì vẫn chia theo số batch
        val_dice_avg = sum_dice / total_val_samples
        val_iou_avg = sum_iou / total_val_samples
        val_pre_avg = sum_pre / total_val_samples
        val_rec_avg = sum_rec / total_val_samples

        # --- PHA 3: LƯU TRỌNG SỐ & IN LOG KẾT QUẢ ---
        torch.save(model.state_dict(), "checkpoints/att_unet_last.pth")
        
        log_str = (f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | "
                   f"Val Loss: {val_loss_avg:.4f} | Dice: {val_dice_avg:.4f} | "
                   f"IoU: {val_iou_avg:.4f} | Pre: {val_pre_avg:.4f} | Rec: {val_rec_avg:.4f}")
        
        if val_dice_avg > best_val_dice:
            best_val_dice = val_dice_avg
            torch.save(model.state_dict(), "checkpoints/att_unet_best.pth")
            log_str = "🥇 [BEST] " + log_str
            
        logger.info(log_str)

if __name__ == "__main__":
    main()