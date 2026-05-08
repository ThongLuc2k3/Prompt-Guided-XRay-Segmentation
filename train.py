import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import Dataset của bạn
from dataset import BTXRD_Dataset

# Import mô hình Attention U-Net từ source code của tác giả
# Lưu ý: Tuỳ thuộc vào cách tác giả đặt tên class, thường là unet_2D
from models.networks.unet_2D import unet_2D

# --- 1. HÀM TÍNH DICE LOSS (Rất quan trọng cho phân đoạn y tế) ---
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred) # Đưa giá trị về khoảng [0, 1]
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# --- 2. CẤU HÌNH HUẤN LUYỆN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
IMG_SIZE = 256

def main():
    print(f"Đang sử dụng thiết bị: {DEVICE}")

    # Khởi tạo Dataloader
    train_dataset = BTXRD_Dataset(image_dir="dataset_BTXRD/train/images", 
                                  mask_dir="dataset_BTXRD/train/masks", 
                                  img_size=IMG_SIZE, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Khởi tạo Mô hình Baseline (Attention U-Net)
    # Ảnh X-quang xám -> in_channels=1, Mask nhị phân -> n_classes=1
    model = unet_2D(in_channels=1, n_classes=1).to(DEVICE)

    # Khởi tạo hàm Loss và Optimizer
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Tạo thư mục lưu trọng số
    os.makedirs("checkpoints", exist_ok=True)

    # --- 3. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP) ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            predictions = model(images)
            
            # Tính tổng Loss = BCE + Dice
            loss_bce = criterion_bce(predictions, masks)
            loss_dice = dice_loss(predictions, masks)
            loss = loss_bce + loss_dice

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"-> Trung bình Loss Epoch {epoch+1}: {epoch_loss/len(train_loader):.4f}")

        # Lưu checkpoint sau mỗi 10 epoch
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/att_unet_epoch_{epoch+1}.pth")
            print("Đã lưu trọng số mô hình!")

if __name__ == "__main__":
    main()