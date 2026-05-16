import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random


class BTXRD_Dataset(Dataset):
    """
    Mỗi sample = 1 polygon GT trong 1 ảnh.
    1 ảnh nhiều polygon → nhiều sample (1GT-1Prompt).

    prompt_mode:
        'zoom_out'   – prompt bao trọn GT, mở rộng ra ngoài
        'shift'      – zoom_out + dịch tâm (vẫn giao GT ≥ 30%)
        'mixed_7_3'  – 70% zoom_out + 30% shift (train: random, test: deterministic by idx)
    """

    def __init__(self, image_dir, json_dir, img_size=512, is_train=True,
                 prompt_mode='zoom_out',
                 zoom_ratio=(0.15, 0.45),
                 shift_ratio=0.30):
        self.image_dir   = image_dir
        self.json_dir    = json_dir
        self.img_size    = img_size
        self.is_train    = is_train
        self.prompt_mode = prompt_mode
        self.zoom_ratio  = zoom_ratio
        self.shift_ratio = shift_ratio

        self.all_samples = []
        for img_name in sorted(os.listdir(image_dir)):
            if not img_name.endswith(('.png', '.jpg')):
                continue
            base = os.path.splitext(img_name)[0]
            json_path = os.path.join(json_dir, base + '.json')
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for i, s in enumerate(data.get('shapes', [])):
                if s.get('shape_type') == 'polygon':
                    self.all_samples.append((img_name, i))

    def __len__(self):
        return len(self.all_samples)

    # ── Prompt helpers ────────────────────────────────────────────────

    def _zoom_out_bbox(self, x_min, x_max, y_min, y_max, orig_h, orig_w):
        """Mở rộng bbox đều ra ngoài GT. Train: asymmetric random. Test: fixed."""
        gt_w, gt_h = x_max - x_min, y_max - y_min
        lo, hi = self.zoom_ratio
        if self.is_train:
            r_l, r_r = random.uniform(lo, hi), random.uniform(lo, hi)
            r_t, r_b = random.uniform(lo, hi), random.uniform(lo, hi)
        else:
            r = (lo + hi) / 2
            r_l = r_r = r_t = r_b = r
        bx_min = max(0,       x_min - gt_w * r_l)
        bx_max = min(orig_w,  x_max + gt_w * r_r)
        by_min = max(0,       y_min - gt_h * r_t)
        by_max = min(orig_h,  y_max + gt_h * r_b)
        return bx_min, bx_max, by_min, by_max

    def _shift_bbox(self, x_min, x_max, y_min, y_max, orig_h, orig_w, seed_idx=None):
        """Zoom-out rồi dịch tâm, đảm bảo overlap với GT ≥ 30%."""
        gt_w, gt_h = x_max - x_min, y_max - y_min
        bx_min, bx_max, by_min, by_max = self._zoom_out_bbox(
            x_min, x_max, y_min, y_max, orig_h, orig_w)

        if self.is_train:
            dx = random.uniform(-gt_w * self.shift_ratio, gt_w * self.shift_ratio)
            dy = random.uniform(-gt_h * self.shift_ratio, gt_h * self.shift_ratio)
        else:
            rng = random.Random(seed_idx or 0)
            dx = rng.uniform(gt_w * 0.4, gt_w * 0.7) * self.shift_ratio
            dy = rng.uniform(gt_h * 0.1, gt_h * 0.3) * self.shift_ratio

        bx_min = max(0,       bx_min + dx)
        bx_max = min(orig_w,  bx_max + dx)
        by_min = max(0,       by_min + dy)
        by_max = min(orig_h,  by_max + dy)

        # Đảm bảo overlap ≥ 30%
        if min(bx_max, x_max) - max(bx_min, x_min) < gt_w * 0.3:
            if dx > 0: bx_max = min(orig_w, x_max + gt_w * 0.15)
            else:      bx_min = max(0,      x_min - gt_w * 0.15)
        if min(by_max, y_max) - max(by_min, y_min) < gt_h * 0.3:
            if dy > 0: by_max = min(orig_h, y_max + gt_h * 0.15)
            else:      by_min = max(0,      y_min - gt_h * 0.15)

        return bx_min, bx_max, by_min, by_max

    def create_plateau_heatmap(self, bbox, orig_h, orig_w):
        heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, int(x_min - 5))
        y_min = max(0, int(y_min - 5))
        x_max = min(orig_w, int(x_max + 5))
        y_max = min(orig_h, int(y_max + 5))
        if x_max > x_min and y_max > y_min:
            heatmap[y_min:y_max, x_min:x_max] = 1.0
            heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
        return heatmap

    # ── Main ──────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        img_name, shape_idx = self.all_samples[idx]
        base = os.path.splitext(img_name)[0]

        image = cv2.imread(os.path.join(self.image_dir, img_name), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = image.shape

        with open(os.path.join(self.json_dir, base + '.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        points = np.array(data['shapes'][shape_idx]['points'])

        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)

        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # Chọn prompt bbox theo mode
        if self.prompt_mode == 'zoom_out':
            bx_min, bx_max, by_min, by_max = self._zoom_out_bbox(
                x_min, x_max, y_min, y_max, orig_h, orig_w)

        elif self.prompt_mode == 'shift':
            bx_min, bx_max, by_min, by_max = self._shift_bbox(
                x_min, x_max, y_min, y_max, orig_h, orig_w, seed_idx=idx)

        elif self.prompt_mode == 'mixed_7_3':
            use_shift = (random.random() < 0.3) if self.is_train else (idx % 10 >= 7)
            if use_shift:
                bx_min, bx_max, by_min, by_max = self._shift_bbox(
                    x_min, x_max, y_min, y_max, orig_h, orig_w, seed_idx=idx)
            else:
                bx_min, bx_max, by_min, by_max = self._zoom_out_bbox(
                    x_min, x_max, y_min, y_max, orig_h, orig_w)
        else:
            raise ValueError(f"Unknown prompt_mode: {self.prompt_mode}")

        prompt_map = self.create_plateau_heatmap(
            [bx_min, by_min, bx_max, by_max], orig_h, orig_w)

        # Resize & normalize
        image      = cv2.resize(image, (self.img_size, self.img_size))
        mask       = cv2.resize(mask, (self.img_size, self.img_size),
                                interpolation=cv2.INTER_NEAREST)
        prompt_map = cv2.resize(prompt_map, (self.img_size, self.img_size))

        image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        mask  = (mask > 127).astype(np.float32)

        image  = torch.from_numpy(image).unsqueeze(0)
        mask   = torch.from_numpy(mask).unsqueeze(0)
        prompt = torch.from_numpy(prompt_map).unsqueeze(0)

        # Augmentation đồng bộ (chỉ train)
        if self.is_train:
            if random.random() >= 0.5:
                image, mask, prompt = TF.hflip(image), TF.hflip(mask), TF.hflip(prompt)
            if random.random() >= 0.5:
                angle  = random.uniform(-15, 15)
                image  = TF.rotate(image,  angle, interpolation=InterpolationMode.BILINEAR)
                mask   = TF.rotate(mask,   angle, interpolation=InterpolationMode.NEAREST)
                prompt = TF.rotate(prompt, angle, interpolation=InterpolationMode.BILINEAR)

        mask = (mask > 0.5).float()
        return image, mask, prompt