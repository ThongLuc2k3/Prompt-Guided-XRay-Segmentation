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
        self.json_dir = json_dir  # Thư mục mới chứa annotations
        self.img_size = img_size
        self.is_train = is_train
        
        # Chỉ lấy các file ảnh hợp lệ
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def create_plateau_heatmap(self, bbox, orig_h, orig_w):
        """
        Hàm vẽ Plateau Heatmap: Mức 1.0 ở trong Box, mờ dần ra viền.
        """
        heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
        
        x_min, y_min, x_max, y_max = bbox
        
        # Đệm thêm 5 pixel (Padding) để Box ôm trọn khối U hơn
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(orig_w, x_max + padding)
        y_max = min(orig_h, y_max + padding)

        # Đổ màu 1.0 (trắng) cho vùng bên trong Box
        heatmap[int(y_min):int(y_max), int(x_min):int(x_max)] = 1.0

        # Làm mờ viền (Gaussian Blur) với kernel lớn
        # Kernel 31x31 giúp tạo vùng mờ (soft-edge) tự nhiên
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
        
        return heatmap

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        json_path = os.path.join(self.json_dir, base_name + '.json') # Giả định file json cùng tên

        # 1. Đọc Ảnh và Mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = image.shape

        # 2. Đọc JSON và tạo Prompt Heatmap
        prompt_map = np.zeros((orig_h, orig_w), dtype=np.float32)
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Quét qua tất cả các hình vẽ trong file LabelMe
                for shape in data.get('shapes', []):
                    # Chỉ lấy hình chữ nhật (Bounding Box)
                    if shape['shape_type'] == 'rectangle':
                        points = shape['points']
                        # points có dạng: [[x1, y1], [x2, y2]]
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        
                        # Đề phòng lúc dùng chuột vẽ bị ngược hướng
                        x_min, x_max = min(x1, x2), max(x1, x2)
                        y_min, y_max = min(y1, y2), max(y1, y2)
                        
                        bbox = [x_min, y_min, x_max, y_max]
                        
                        # Tạo heatmap cho Box này
                        single_heatmap = self.create_plateau_heatmap(bbox, orig_h, orig_w)
                        
                        # Cộng dồn vào bản đồ tổng (Dùng np.maximum để không bị vượt quá 1.0 nếu 2 box đè lên nhau)
                        prompt_map = np.maximum(prompt_map, single_heatmap)
        else:
            # Nếu lỡ có ảnh không có file json, prompt mặc định là 0 (Không định hướng)
            pass

        # 3. Resize đồng loạt về kích thước chuẩn
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        prompt_map = cv2.resize(prompt_map, (self.img_size, self.img_size))

        # 4. Chuẩn hóa (Normalize)
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask[mask > 0.5] = 1.0 
        mask[mask <= 0.5] = 0.0
        
        # Heatmap bản chất đã từ 0 -> 1 sau khi qua Gaussian Blur, không cần chia 255

        # 5. Chuyển sang Tensor [C, H, W]
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        prompt = torch.from_numpy(prompt_map).unsqueeze(0)

        # 6. Data Augmentation (Dùng lật ngang đồng thời cho cả 3)
        if self.is_train and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            prompt = TF.hflip(prompt)

        # Output bây giờ trả về TẬN 3 TENSOR!
        return image, mask, prompt

# --- TEST THỬ DATALOADER ---
if __name__ == "__main__":
    train_dataset = BTXRD_Dataset(
        image_dir="dataset_BTXRD/train/images", 
        mask_dir="dataset_BTXRD/train/masks", 
        json_dir="dataset_BTXRD/train/annotations", # Thư mục JSON mới
        is_train=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    images, masks, prompts = next(iter(train_loader))
    
    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)
    print("Prompt batch shape:", prompts.shape) # Phải là [4, 1, 256, 256 or 512..]
    print(f"Giá trị đỉnh của một Prompt: {prompts[0].max().item():.2f}")