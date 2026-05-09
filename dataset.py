import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random

class BTXRD_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, json_dir, img_size=512, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.json_dir = json_dir  
        self.img_size = img_size
        self.is_train = is_train
        
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def create_plateau_heatmap(self, bbox, orig_h, orig_w):
        """Hàm vẽ Plateau Heatmap: Mức 1.0 ở trong Box, mờ dần ra viền."""
        heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
        x_min, y_min, x_max, y_max = bbox
        
        padding = 5
        x_min = max(0, int(x_min - padding))
        y_min = max(0, int(y_min - padding))
        x_max = min(orig_w, int(x_max + padding))
        y_max = min(orig_h, int(y_max + padding))

        heatmap[y_min:y_max, x_min:x_max] = 1.0
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
        return heatmap

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        json_path = os.path.join(self.json_dir, base_name + '.json')

        # 1. Đọc Ảnh và Mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = image.shape

        # 2. Đọc JSON và tạo Prompt Heatmap (CHỐNG HỌC VỆT)
        prompt_map = np.zeros((orig_h, orig_w), dtype=np.float32)
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for shape in data.get('shapes', []):
                    if shape['shape_type'] == 'rectangle':
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        
                        x_min, x_max = min(x1, x2), max(x1, x2)
                        y_min, y_max = min(y1, y2), max(y1, y2)
                        
                        # 🔥 ĐÒN 1: ZOOM OUT (90%) & ZOOM IN (10%) 🔥
                        if self.is_train:
                            if random.random() < 0.9:
                                # Zoom Out (Mở rộng box, ép tự gọt vào)
                                pad_l, pad_r = random.randint(5, 30), random.randint(5, 30)
                                pad_t, pad_b = random.randint(5, 30), random.randint(5, 30)
                                x_min, y_min = x_min - pad_l, y_min - pad_t
                                x_max, y_max = x_max + pad_r, y_max + pad_b
                            else:
                                # Zoom In (Bóp nhỏ box, ép tự loang ra)
                                pad_l, pad_r = random.randint(5, 15), random.randint(5, 15)
                                pad_t, pad_b = random.randint(5, 15), random.randint(5, 15)
                                x_min, y_min = x_min + pad_l, y_min + pad_t
                                x_max, y_max = x_max - pad_r, y_max - pad_b
                                
                                # Đảm bảo Zoom In không làm box bị lật ngược tọa độ
                                if x_min >= x_max: x_min, x_max = x_max - 1, x_min + 1
                                if y_min >= y_max: y_min, y_max = y_max - 1, y_min + 1

                        # Giới hạn tọa độ không vượt quá ảnh
                        x_min, y_min = max(0, x_min), max(0, y_min)
                        x_max, y_max = min(orig_w, x_max), min(orig_h, y_max)
                        
                        bbox = [x_min, y_min, x_max, y_max]
                        single_heatmap = self.create_plateau_heatmap(bbox, orig_h, orig_w)
                        prompt_map = np.maximum(prompt_map, single_heatmap)
                        
        # 🔥 ĐÒN 2: BẪY PROMPT GIẢ (10% cơ hội) 🔥
        if self.is_train and random.random() < 0.10:
            fake_w, fake_h = random.randint(30, 100), random.randint(30, 100)
            fake_x = random.randint(0, max(1, orig_w - fake_w))
            fake_y = random.randint(0, max(1, orig_h - fake_h))
            fake_heatmap = self.create_plateau_heatmap([fake_x, fake_y, fake_x + fake_w, fake_y + fake_h], orig_h, orig_w)
            prompt_map = np.maximum(prompt_map, fake_heatmap)

        # 🔥 ĐÒN 3: PROMPT DROPOUT (20% cơ hội) 🔥
        if self.is_train and random.random() < 0.20:
            prompt_map = np.zeros((orig_h, orig_w), dtype=np.float32)

        # 3. Resize đồng loạt về kích thước chuẩn
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        prompt_map = cv2.resize(prompt_map, (self.img_size, self.img_size))

        # 4. Chuẩn hóa (Normalize)
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask[mask > 0.5] = 1.0 
        mask[mask <= 0.5] = 0.0

        # 5. Chuyển sang Tensor [C, H, W]
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        prompt = torch.from_numpy(prompt_map).unsqueeze(0)

        # ==========================================
        # 🔥 ĐÒN 4: ĐỒNG BỘ HÌNH HỌC (DATA AUGMENTATION) 🔥
        # Xử lý trên Tensor để đảm bảo Ảnh, Mask, Prompt ăn khớp 100%
        # ==========================================
        if self.is_train:
            # 4.1 Lật ngang ngẫu nhiên (50%)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                prompt = TF.hflip(prompt)
                
            # 4.2 Xoay nghiêng ngẫu nhiên từ -15 đến +15 độ (50%)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
                prompt = TF.rotate(prompt, angle)

        return image, mask, prompt

# --- TEST THỬ DATALOADER ---
if __name__ == "__main__":
    train_dataset = BTXRD_Dataset(
        image_dir="dataset_BTXRD/train/images", 
        mask_dir="dataset_BTXRD/train/masks", 
        json_dir="dataset_BTXRD/train/annotations",
        is_train=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    images, masks, prompts = next(iter(train_loader))
    
    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)
    print("Prompt batch shape:", prompts.shape)
    print(f"Giá trị đỉnh của một Prompt: {prompts[0].max().item():.2f}")