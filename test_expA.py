"""
TEST – Thí nghiệm A: Zoom-out only
Đánh giá trên tập test với prompt cố định (zoom-out, fixed ratio).
6 metrics: Dice, IoU, Precision, Recall, HD95, CBL
"""

import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import binary_erosion, distance_transform_edt

from dataset import BTXRD_Dataset
from models.networks.prompt_unet_2D import PGA_UNet

# ── Cấu hình ──────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/pga_unet_expA_best.pth"
IMG_SIZE   = 512
BATCH_SIZE = 4

TEST_IMAGE_DIR = "dataset_BTXRD/test/images"
TEST_JSON_DIR  = "dataset_BTXRD/test/annotations"

# ── Metrics helpers ───────────────────────────────────────────────────

def calc_hd95(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(IMG_SIZE)
    pe = pred ^ binary_erosion(pred)
    ge = gt   ^ binary_erosion(gt)
    dt_p = distance_transform_edt(~pe)
    dt_g = distance_transform_edt(~ge)
    d1 = dt_g[pe]
    d2 = dt_p[ge]
    if len(d1) == 0 or len(d2) == 0:
        return float(IMG_SIZE)
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def calc_cbl_numpy(pred_bin, gt_bin):
    """CBL cho numpy arrays (H, W) binary."""
    gt_area = gt_bin.sum()
    if gt_area == 0:
        return None  # skip
    ys, xs = np.where(gt_bin)
    cx_gt  = xs.mean()
    cy_gt  = ys.mean()
    gt_diag = np.sqrt((ys.max() - ys.min()) ** 2 + (xs.max() - xs.min()) ** 2) + 1e-6

    pred_area = pred_bin.sum()
    if pred_area == 0:
        return 0.0
    ys_p, xs_p = np.where(pred_bin)
    cx_p = xs_p.mean()
    cy_p = ys_p.mean()
    d = np.sqrt((cx_p - cx_gt) ** 2 + (cy_p - cy_gt) ** 2)
    return float(np.clip(1.0 - d / gt_diag, 0, 1))


# ── Đánh giá toàn bộ test set ─────────────────────────────────────────

def evaluate():
    print("\n" + "=" * 70)
    print("TEST – Thí nghiệm A: Zoom-out only")
    print("=" * 70)

    model = PGA_UNet(in_channels=1, n_classes=1, use_encoder_prompt=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    test_ds = BTXRD_Dataset(
        image_dir=TEST_IMAGE_DIR, json_dir=TEST_JSON_DIR,
        img_size=IMG_SIZE, is_train=False,
        prompt_mode='zoom_out'
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_dice, all_iou, all_pre, all_rec, all_hd95, all_cbl = [], [], [], [], [], []
    smooth = 1e-5

    with torch.no_grad():
        for images, masks, prompts in test_loader:
            images  = images.to(DEVICE)
            masks   = masks.to(DEVICE)
            prompts = prompts.to(DEVICE)
            preds   = model(images, prompts)
            preds_b = (torch.sigmoid(preds) > 0.5).float()

            for b in range(images.size(0)):
                pm = preds_b[b, 0].cpu().numpy()
                gm = masks[b, 0].cpu().numpy()

                tp = (pm * gm).sum()
                fp = (pm * (1 - gm)).sum()
                fn = ((1 - pm) * gm).sum()

                all_dice.append((2*tp + smooth) / (2*tp + fp + fn + smooth))
                all_iou.append((tp + smooth) / (tp + fp + fn + smooth))
                all_pre.append((tp + smooth) / (tp + fp + smooth))
                all_rec.append((tp + smooth) / (tp + fn + smooth))
                all_hd95.append(calc_hd95(pm, gm))

                cbl = calc_cbl_numpy(pm.astype(bool), gm.astype(bool))
                if cbl is not None:
                    all_cbl.append(cbl)

    print(f"\n{'Metric':<12} {'Giá trị':>10}")
    print("-" * 25)
    print(f"{'Dice ↑':<12} {np.mean(all_dice):>10.4f}")
    print(f"{'IoU ↑':<12} {np.mean(all_iou):>10.4f}")
    print(f"{'Precision ↑':<12} {np.mean(all_pre):>10.4f}")
    print(f"{'Recall ↑':<12} {np.mean(all_rec):>10.4f}")
    print(f"{'HD95 ↓(px)':<12} {np.mean(all_hd95):>10.2f}")
    print(f"{'CBL ↑':<12} {np.mean(all_cbl):>10.4f}")
    print(f"\nSố sample: {len(all_dice)}")
    return model


# ── Visualize ảnh cụ thể ────────────────────────────────────────────

def visualize(model, img_name):
    img_path  = os.path.join(TEST_IMAGE_DIR, img_name)
    json_path = os.path.join(TEST_JSON_DIR, os.path.splitext(img_name)[0] + '.json')
    if not os.path.exists(img_path):
        print(f"Không tìm thấy: {img_path}")
        return

    image_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = image_cv.shape

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Tự động tìm shape là polygon chuẩn để vẽ
    target_shape = None
    target_idx = 0
    for idx, s in enumerate(data.get('shapes', [])):
        if s.get('shape_type') == 'polygon' and len(s['points']) > 2:
            target_shape = s
            target_idx = idx
            break
            
    if target_shape is None:
        if len(data.get('shapes', [])) > 0:
            target_shape = data['shapes'][0]
        else:
            print(f"File JSON {img_name} không có nhãn nào.")
            return

    print(f"\nVisualize: {img_name} – Đang vẽ vùng xương thực tế (Shape index: {target_idx})")
    points = np.array(target_shape['points']).astype(np.int32)

    mask_cv = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cv2.fillPoly(mask_cv, [points], 255)

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    # Tạo prompt zoom-out cố định (test)
    ds_tmp = BTXRD_Dataset(TEST_IMAGE_DIR, TEST_JSON_DIR, img_size=IMG_SIZE,
                           is_train=False, prompt_mode='zoom_out')
    r = (ds_tmp.zoom_ratio[0] + ds_tmp.zoom_ratio[1]) / 2
    gt_w, gt_h = x_max - x_min, y_max - y_min
    bx_min = max(0,       x_min - gt_w * r)
    bx_max = min(orig_w,  x_max + gt_w * r)
    by_min = max(0,       y_min - gt_h * r)
    by_max = min(orig_h,  y_max + gt_h * r)
    prompt_map = ds_tmp.create_plateau_heatmap([bx_min, by_min, bx_max, by_max], orig_h, orig_w)

    # Resize về 512x512
    image_r  = cv2.resize(image_cv, (IMG_SIZE, IMG_SIZE))
    mask_r   = cv2.resize(mask_cv,  (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    prompt_r = cv2.resize(prompt_map, (IMG_SIZE, IMG_SIZE))

    img_t = torch.from_numpy((image_r.astype(np.float32)/255.0 - 0.5)/0.5).unsqueeze(0).unsqueeze(0).to(DEVICE)
    pmt_t = torch.from_numpy(prompt_r).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_t = (torch.sigmoid(model(img_t, pmt_t)) > 0.5).float()

    img_np   = image_r / 255.0
    gt_np    = (mask_r > 127).astype(np.float32)
    pred_np  = pred_t[0, 0].cpu().numpy()

    # --- TÍNH TỌA ĐỘ TÂM (CENTROID) ---
    # Tâm của Ground Truth
    if gt_np.sum() > 0:
        ys_gt, xs_gt = np.where(gt_np == 1)
        cx_gt, cy_gt = xs_gt.mean(), ys_gt.mean()
    else:
        cx_gt, cy_gt = None, None

    # Tâm của Dự đoán
    if pred_np.sum() > 0:
        ys_pred, xs_pred = np.where(pred_np == 1)
        cx_pred, cy_pred = xs_pred.mean(), ys_pred.mean()
    else:
        cx_pred, cy_pred = None, None

    tp   = (pred_np * gt_np).sum()
    dice = (2*tp + 1e-5) / (pred_np.sum() + gt_np.sum() + 1e-5)
    hd95 = calc_hd95(pred_np.astype(bool), gt_np.astype(bool))
    cbl  = calc_cbl_numpy(pred_np.astype(bool), gt_np.astype(bool))

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Exp A – Zoom-out | {img_name} (Shape {target_idx})", fontsize=14, fontweight='bold')

    titles = ["Ảnh gốc", "Prompt heatmap", "Ground Truth", "Dự đoán",
              f"Biên (Dice:{dice:.3f} HD95:{hd95:.1f}px CBL:{cbl:.3f})"]

    axes[0].imshow(img_np, cmap='gray')
    
    axes[1].imshow(img_np, cmap='gray')
    axes[1].imshow(np.ma.masked_where(prompt_r < 0.05, prompt_r), cmap='magma', alpha=0.6)
    
    # --- Ô 3: Ground Truth ---
    axes[2].imshow(img_np, cmap='gray')
    green_overlay = np.zeros((*gt_np.shape, 4))
    green_overlay[gt_np == 1] = [0, 1, 0, 0.35] 
    axes[2].imshow(green_overlay)
    if gt_np.max() > 0:
        axes[2].contour(gt_np, levels=[0.5], colors='lime', linewidths=1.5)
    # Vẽ chấm tâm GT màu xanh lá sáng (alpha=1)
    if cx_gt is not None:
        axes[2].plot(cx_gt, cy_gt, marker='o', color='lime', markersize=8, alpha=1.0, markeredgecolor='black')
        
    # --- Ô 4: Dự đoán ---
    axes[3].imshow(img_np, cmap='gray')
    red_overlay = np.zeros((*pred_np.shape, 4))
    red_overlay[pred_np == 1] = [1, 0, 0, 0.35] 
    axes[3].imshow(red_overlay)
    # Vẽ chấm tâm Dự đoán màu đỏ sáng (alpha=1)
    if cx_pred is not None:
        axes[3].plot(cx_pred, cy_pred, marker='o', color='red', markersize=8, alpha=1.0, markeredgecolor='white')
    
    # --- Ô 5: Biên so sánh độ lệch tâm ---
    axes[4].imshow(img_np, cmap='gray')
    if gt_np.max() > 0:   axes[4].contour(gt_np,   levels=[0.5], colors='lime',  linewidths=2)
    if pred_np.max() > 0: axes[4].contour(pred_np, levels=[0.5], colors='red',   linewidths=2, linestyles='dashed')
    
    # Vẽ cả 2 chấm tâm lên ô so sánh để nhìn rõ khoảng cách lệch
    if cx_gt is not None:
        axes[4].plot(cx_gt, cy_gt, marker='o', color='lime', markersize=8, alpha=1.0, markeredgecolor='black', label='Tâm GT')
    if cx_pred is not None:
        axes[4].plot(cx_pred, cy_pred, marker='o', color='red', markersize=8, alpha=1.0, markeredgecolor='white', label='Tâm Pred')

    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model = evaluate()
    
    # Danh sách các ảnh cần chạy visualize
    list_images = [
        "IMG001768.png",
        "IMG001538.png",
        "IMG001100.png",
        "IMG001397.png",
        "IMG000235.png"
    ]
    
    for img in list_images:
        visualize(trained_model, img_name=img) # Bỏ shape_idx cứng, để hàm tự tìm polygon