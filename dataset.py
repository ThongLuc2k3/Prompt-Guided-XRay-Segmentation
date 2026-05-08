import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random

class BTXRD_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Đường dẫn tới ảnh và mask
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) # Giả định file mask cùng tên

        # Đọc ảnh X-quang (ảnh xám)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Đọc mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize về kích thước chuẩn
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Chuẩn hóa (Normalize)
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask[mask > 0.5] = 1.0 # Đảm bảo mask nhị phân
        mask[mask <= 0.5] = 0.0

        # Chuyển sang Tensor [C, H, W]
        image = torch.from_numpy(image).unsqueeze(0) # [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)   # [1, H, W]

        # Data Augmentation đơn giản cho tập Train (Random Lật ngang)
        if self.is_train and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask

# Kiểm tra thử Dataloader
if __name__ == "__main__":
    # Nhớ đổi đường dẫn này trúng với máy của bạn nhé
    train_dataset = BTXRD_Dataset(image_dir="dataset_BTXRD/train/images", 
                                  mask_dir="dataset_BTXRD/train/masks", 
                                  is_train=True)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    images, masks = next(iter(train_loader))
    print("Image batch shape:", images.shape) # Output mong muốn: [4, 1, 256, 256]
    print("Mask batch shape:", masks.shape)