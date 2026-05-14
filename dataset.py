import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random

class BTXRD_Dataset(Dataset):
    def __init__(self, image_dir, json_dir, img_size=512, is_train=True):
        self.image_dir = image_dir
        self.json_dir = json_dir  
        self.img_size = img_size
        self.is_train = is_train
        
        self.all_samples = []

        img_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
        for img_name in img_files:
            base_name = os.path.splitext(img_name)[0]
            json_path = os.path.join(json_dir, base_name + '.json')
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    polygon_shapes = [i for i, s in enumerate(data.get('shapes', [])) 
                                    if s.get('shape_type') == 'polygon']
                    for shape_idx in polygon_shapes:
                        self.all_samples.append((img_name, shape_idx))

    def __len__(self):
        return len(self.all_samples)

    def create_plateau_heatmap(self, bbox, orig_h, orig_w):
        heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
        x_min, y_min, x_max, y_max = bbox
        
        padding = 5
        x_min, y_min = max(0, int(x_min - padding)), max(0, int(y_min - padding))
        x_max, y_max = min(orig_w, int(x_max + padding)), min(orig_h, int(y_max + padding))

        heatmap[y_min:y_max, x_min:x_max] = 1.0
        return cv2.GaussianBlur(heatmap, (31, 31), 0)

    def __getitem__(self, idx):
        img_name, shape_idx = self.all_samples[idx]
        base_name = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(self.image_dir, img_name)
        json_path = os.path.join(self.json_dir, base_name + '.json')

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = image.shape

        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        prompt_map = np.zeros((orig_h, orig_w), dtype=np.float32)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            points = np.array(data['shapes'][shape_idx]['points'])

            pts = points.astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
                        
            if self.is_train:
                # ĐÒN 1: ZOOM
                gt_w, gt_h = x_max - x_min, y_max - y_min
                ratio_w = random.uniform(0.9, 1.2) if gt_w < 100 else random.uniform(0.3, 0.5)
                ratio_h = random.uniform(0.9, 1.2) if gt_h < 100 else random.uniform(0.3, 0.5)

                pad_w, pad_h = gt_w * ratio_w, gt_h * ratio_h
                shift_w, shift_h = random.uniform(0, pad_w), random.uniform(0, pad_h)
                
                x_min -= shift_w
                x_max += (pad_w - shift_w)
                y_min -= shift_h
                y_max += (pad_h - shift_h)

                # Cấp quyền thực hiện các đòn nguy hiểm
                apply_other_tricks = random.random() < 0.5 

                # ĐÒN 2: RANDOM SHIFT (Độc lập xác suất)
                if apply_other_tricks and random.random() < 0.60:
                    side = random.choice(["left", "right", "top", "bottom"])
                    cut_ratio = random.uniform(0.1, 0.3)
                    if side == "left": x_min += int(gt_w * cut_ratio)
                    elif side == "right": x_max -= int(gt_w * cut_ratio)
                    elif side == "top": y_min += int(gt_h * cut_ratio)
                    elif side == "bottom": y_max -= int(gt_h * cut_ratio)

            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(orig_w, int(x_max)), min(orig_h, int(y_max))
            
            single_heatmap = self.create_plateau_heatmap([x_min, y_min, x_max, y_max], orig_h, orig_w)
            prompt_map = np.maximum(prompt_map, single_heatmap)
                        
            # ĐÒN 3: BẪY PROMPT GIẢ (Độc lập xác suất)
            if self.is_train and apply_other_tricks and random.random() < 0.30:
                fake_w, fake_h = random.randint(100, 200), random.randint(100, 200)
                fake_x = random.randint(0, max(1, orig_w - fake_w))
                fake_y = random.randint(0, max(1, orig_h - fake_h))
                fake_heatmap = self.create_plateau_heatmap([fake_x, fake_y, fake_x + fake_w, fake_y + fake_h], orig_h, orig_w)
                prompt_map = np.maximum(prompt_map, fake_heatmap)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        prompt_map = cv2.resize(prompt_map, (self.img_size, self.img_size))

        # Chuẩn hóa Z-score cho ảnh X-quang (Mean/Std có thể điều chỉnh theo tập data thực tế)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5 
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        prompt = torch.from_numpy(prompt_map).unsqueeze(0)

        # ĐÒN 5: ĐỒNG BỘ HÌNH HỌC 
        if self.is_train:
            if random.random() >= 0.5:
                image, mask, prompt = TF.hflip(image), TF.hflip(mask), TF.hflip(prompt)
                
            if random.random() >= 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                prompt = TF.rotate(prompt, angle, interpolation=InterpolationMode.BILINEAR)
                
        mask = (mask > 0.5).float()
        return image, mask, prompt