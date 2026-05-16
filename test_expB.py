"""
TEST – Thí nghiệm B: Zoom-out + Shift (+ Cải tiến inference)
Đánh giá 3 kịch bản: 100% zoom-out, 100% shift, mixed 70-30.
Thêm hàm phát hiện prompt sai + GradCAM gợi ý vùng u.
"""

import os
import cv2
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import binary_erosion, distance_transform_edt

from dataset import BTXRD_Dataset
from models.networks.prompt_unet_2D import PGA_UNet

# ── Cấu hình ──────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/pga_unet_expB_best.pth"
IMG_SIZE   = 512
BATCH_SIZE = 4

TEST_IMAGE_DIR = "dataset_BTXRD/test/images"
TEST_JSON_DIR  = "dataset_BTXRD/test/annotations"

VIZ_IMG_NAME  = "IMG001768.png"
VIZ_SHAPE_IDX = 0

CONFIDENCE_THRESHOLD = 0.3  # Dưới ngưỡng này → nghi ngờ prompt sai


# ── Metrics helpers (giống test_expA) ────────────────────────────────

def calc_hd95(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any(): return 0.0
    if not pred.any() or not gt.any():  return float(IMG_SIZE)
    pe = pred ^ binary_erosion(pred)
    ge = gt   ^ binary_erosion(gt)
    d1 = distance_transform_edt(~ge)[pe]
    d2 = distance_transform_edt(~pe)[ge]
    if not len(d1) or not len(d2): return float(IMG_SIZE)
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def calc_cbl_numpy(pred_bin, gt_bin):
    gt_area = gt_bin.sum()
    if gt_area == 0: return None
    ys, xs = np.where(gt_bin)
    gt_diag = np.sqrt((ys.max()-ys.min())**2 + (xs.max()-xs.min())**2) + 1e-6
    pred_area = pred_bin.sum()
    if pred_area == 0: return 0.0
    yp, xp = np.where(pred_bin)
    d = np.sqrt((xp.mean()-xs.mean())**2 + (yp.mean()-ys.mean())**2)
    return float(np.clip(1.0 - d/gt_diag, 0, 1))


def run_metrics(model, loader):
    """Chạy eval cho 1 loader, trả về dict các metric trung bình."""
    all_dice, all_iou, all_pre, all_rec, all_hd95, all_cbl = [], [], [], [], [], []
    smooth = 1e-5
    model.eval()
    with torch.no_grad():
        for images, masks, prompts in loader:
            images, masks, prompts = (images.to(DEVICE), masks.to(DEVICE), prompts.to(DEVICE))
            preds = (torch.sigmoid(model(images, prompts)) > 0.5).float()
            for b in range(images.size(0)):
                pm = preds[b, 0].cpu().numpy()
                gm = masks[b, 0].cpu().numpy()
                tp = (pm*gm).sum(); fp = (pm*(1-gm)).sum(); fn = ((1-pm)*gm).sum()
                all_dice.append((2*tp+smooth)/(2*tp+fp+fn+smooth))
                all_iou.append((tp+smooth)/(tp+fp+fn+smooth))
                all_pre.append((tp+smooth)/(tp+fp+smooth))
                all_rec.append((tp+smooth)/(tp+fn+smooth))
                all_hd95.append(calc_hd95(pm.astype(bool), gm.astype(bool)))
                cbl = calc_cbl_numpy(pm.astype(bool), gm.astype(bool))
                if cbl is not None: all_cbl.append(cbl)
    return {
        'dice': np.mean(all_dice), 'iou':  np.mean(all_iou),
        'pre':  np.mean(all_pre),  'rec':  np.mean(all_rec),
        'hd95': np.mean(all_hd95), 'cbl':  np.mean(all_cbl) if all_cbl else 0.0,
        'n':    len(all_dice)
    }


# ── Đánh giá 3 kịch bản ──────────────────────────────────────────────

def evaluate():
    print("\n" + "=" * 75)
    print("TEST – Thí nghiệm B: 3 kịch bản prompt")
    print("=" * 75)

    model = PGA_UNet(in_channels=1, n_classes=1, use_encoder_prompt=True).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    scenarios = {
        '100% Zoom-out': 'zoom_out',
        '100% Shift':    'shift',
        'Mixed 70-30':   'mixed_7_3',
    }

    results = {}
    for name, mode in scenarios.items():
        ds = BTXRD_Dataset(
            image_dir=TEST_IMAGE_DIR, json_dir=TEST_JSON_DIR,
            img_size=IMG_SIZE, is_train=False, prompt_mode=mode
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        results[name] = run_metrics(model, loader)
        print(f"[{name}] done – {results[name]['n']} samples")

    # In bảng kết quả
    print(f"\n{'Kịch bản':<16} {'Dice↑':>7} {'IoU↑':>7} {'Pre↑':>7} {'Rec↑':>7} {'HD95↓':>8} {'CBL↑':>7} {'N':>5}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<16} {r['dice']:>7.4f} {r['iou']:>7.4f} {r['pre']:>7.4f} "
              f"{r['rec']:>7.4f} {r['hd95']:>8.2f} {r['cbl']:>7.4f} {r['n']:>5}")

    return model


# ── Visualize 1 ảnh với 3 kịch bản ──────────────────────────────────

def visualize_3_scenarios(model, img_name=VIZ_IMG_NAME, shape_idx=VIZ_SHAPE_IDX):
    print(f"\nVisualize 3 kịch bản: {img_name} – shape {shape_idx}")

    img_path  = os.path.join(TEST_IMAGE_DIR, img_name)
    json_path = os.path.join(TEST_JSON_DIR, os.path.splitext(img_name)[0] + '.json')
    if not os.path.exists(img_path):
        print(f"Không tìm thấy: {img_path}")
        return

    image_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = image_cv.shape
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    points = np.array(data['shapes'][shape_idx]['points'])

    mask_cv = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cv2.fillPoly(mask_cv, [points.astype(np.int32)], 255)

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    gt_w, gt_h   = x_max - x_min, y_max - y_min

    # Tạo dataset helper để dùng create_plateau_heatmap
    ds_helper = BTXRD_Dataset(TEST_IMAGE_DIR, TEST_JSON_DIR,
                               img_size=IMG_SIZE, is_train=False)
    r_fixed = (ds_helper.zoom_ratio[0] + ds_helper.zoom_ratio[1]) / 2

    def make_prompt(bx_min, bx_max, by_min, by_max):
        pm = ds_helper.create_plateau_heatmap([bx_min, by_min, bx_max, by_max], orig_h, orig_w)
        return cv2.resize(pm, (IMG_SIZE, IMG_SIZE))

    prompts_np = {
        'Zoom-out': make_prompt(
            x_min - gt_w*r_fixed, x_max + gt_w*r_fixed,
            y_min - gt_h*r_fixed, y_max + gt_h*r_fixed
        ),
        'Shift': make_prompt(
            x_min + gt_w*0.3, x_max + gt_w*0.3 + gt_w*(1+r_fixed),
            y_min + gt_h*0.2, y_max + gt_h*0.2 + gt_h*(1+r_fixed)
        ),
        'Mixed\n(shift ex.)': make_prompt(   # minh họa phần shift của mixed
            x_min - gt_w*0.1, x_max + gt_w*0.5,
            y_min + gt_h*0.2, y_max + gt_h*0.8
        ),
    }

    image_r = cv2.resize(image_cv, (IMG_SIZE, IMG_SIZE))
    mask_r  = (cv2.resize(mask_cv, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 127)
    img_t   = torch.from_numpy((image_r.astype(np.float32)/255.0-0.5)/0.5).unsqueeze(0).unsqueeze(0).to(DEVICE)
    img_np  = image_r / 255.0
    gt_np   = mask_r.astype(np.float32)

    fig, axes = plt.subplots(3, 5, figsize=(24, 14))
    fig.suptitle(f"Exp B – 3 kịch bản prompt | {img_name} shape {shape_idx}",
                 fontsize=15, fontweight='bold')

    col_titles = ["Ảnh gốc", "Prompt", "Ground Truth", "Dự đoán", "Biên (Dice | CBL)"]
    for c, t in enumerate(col_titles):
        axes[0][c].set_title(t, fontsize=11, fontweight='bold')

    model.eval()
    with torch.no_grad():
        for row, (scenario_name, pm_np) in enumerate(prompts_np.items()):
            pm_t = torch.from_numpy(pm_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred_np = (torch.sigmoid(model(img_t, pm_t)) > 0.5).float()[0, 0].cpu().numpy()

            tp   = (pred_np * gt_np).sum()
            dice = (2*tp + 1e-5) / (pred_np.sum() + gt_np.sum() + 1e-5)
            hd95 = calc_hd95(pred_np.astype(bool), gt_np.astype(bool))
            cbl  = calc_cbl_numpy(pred_np.astype(bool), gt_np.astype(bool)) or 0.0

            axes[row][0].imshow(img_np, cmap='gray')
            axes[row][0].set_ylabel(scenario_name, fontsize=11, fontweight='bold', rotation=0,
                                    labelpad=60, va='center')

            axes[row][1].imshow(img_np, cmap='gray')
            axes[row][1].imshow(np.ma.masked_where(pm_np < 0.05, pm_np), cmap='magma', alpha=0.6)

            axes[row][2].imshow(img_np, cmap='gray')
            axes[row][2].imshow(np.ma.masked_where(gt_np == 0, gt_np), cmap='Greens', alpha=0.5)

            axes[row][3].imshow(img_np, cmap='gray')
            axes[row][3].imshow(np.ma.masked_where(pred_np == 0, pred_np), cmap='Reds', alpha=0.5)

            axes[row][4].imshow(img_np, cmap='gray')
            if gt_np.max() > 0:   axes[row][4].contour(gt_np,   [0.5], colors='lime', linewidths=2)
            if pred_np.max() > 0: axes[row][4].contour(pred_np, [0.5], colors='red',  linewidths=2, linestyles='--')
            axes[row][4].set_title(f"Dice:{dice:.3f} | HD95:{hd95:.1f}px | CBL:{cbl:.3f}",
                                   fontsize=9, fontweight='bold')

            for ax in axes[row]:
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"result_expB_{os.path.splitext(img_name)[0]}_shape{shape_idx}.png",
                dpi=150, bbox_inches='tight')
    plt.show()


# ── Cải tiến inference: phát hiện prompt sai + GradCAM ───────────────

def predict_with_check(model, image_tensor, prompt_tensor):
    """
    Chạy inference + phát hiện nếu prompt có khả năng sai vị trí.
    Trả về: mask, confidence, is_suspicious, saliency_map
    """
    model.eval()
    image_tensor  = image_tensor.to(DEVICE)
    prompt_tensor = prompt_tensor.to(DEVICE)

    # Forward chính
    with torch.no_grad():
        out  = model(image_tensor, prompt_tensor)
        prob = torch.sigmoid(out)
        pred = (prob > 0.5).float()

    confidence = prob.max().item()

    # Kiểm tra khoảng cách tâm prompt vs tâm predicted mask
    prompt_np = prompt_tensor[0, 0].cpu().numpy()
    pred_np   = pred[0, 0].cpu().numpy()
    H, W = pred_np.shape

    ys_pmt, xs_pmt = np.where(prompt_np > 0.3)
    cx_pmt = xs_pmt.mean() if len(xs_pmt) > 0 else W / 2
    cy_pmt = ys_pmt.mean() if len(ys_pmt) > 0 else H / 2

    center_dist = float(IMG_SIZE)  # default: xa
    if pred_np.sum() > 0:
        ys_pred, xs_pred = np.where(pred_np > 0)
        cx_pred = xs_pred.mean()
        cy_pred = ys_pred.mean()
        center_dist = np.sqrt((cx_pred - cx_pmt)**2 + (cy_pred - cy_pmt)**2)

    is_suspicious = (confidence < CONFIDENCE_THRESHOLD) or (center_dist > IMG_SIZE * 0.25)

    # GradCAM – chỉ tính khi nghi ngờ prompt sai
    saliency = None
    if is_suspicious:
        saliency = _gradcam(model, image_tensor)

    return {
        'mask':         pred[0, 0].cpu().numpy(),
        'prob_map':     prob[0, 0].cpu().numpy(),
        'confidence':   confidence,
        'center_dist':  center_dist,
        'is_suspicious': is_suspicious,
        'saliency':     saliency,
    }


def _gradcam(model, image_tensor):
    """GradCAM w.r.t. bottleneck (center layer)."""
    gradients, activations = [], []

    def fwd_hook(module, inp, out):
        activations.append(out)
        out.register_hook(lambda g: gradients.append(g))

    hook = model.center.register_forward_hook(fwd_hook)
    model.eval()

    img = image_tensor.clone().to(DEVICE)
    zero_prompt = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
    out = model(img, zero_prompt)
    model.zero_grad()
    out.sum().backward()
    hook.remove()

    if not gradients or not activations:
        return None

    g = gradients[0]            # (1, C, h, w)
    a = activations[0]          # (1, C, h, w)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def visualize_inference_check(model, img_name=VIZ_IMG_NAME, shape_idx=VIZ_SHAPE_IDX):
    """
    Minh họa cải tiến inference:
    Hàng 1 = prompt đúng (zoom-out).
    Hàng 2 = prompt lệch nhiều (giả lập sai vị trí).
    Cột thêm: confidence, saliency (GradCAM gợi ý).
    """
    print(f"\nVisualize inference check: {img_name}")

    img_path  = os.path.join(TEST_IMAGE_DIR, img_name)
    json_path = os.path.join(TEST_JSON_DIR, os.path.splitext(img_name)[0] + '.json')
    if not os.path.exists(img_path): return

    image_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = image_cv.shape
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    points = np.array(data['shapes'][shape_idx]['points'])

    mask_cv = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cv2.fillPoly(mask_cv, [points.astype(np.int32)], 255)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    gt_w, gt_h   = x_max - x_min, y_max - y_min

    ds_h = BTXRD_Dataset(TEST_IMAGE_DIR, TEST_JSON_DIR, img_size=IMG_SIZE, is_train=False)
    r    = (ds_h.zoom_ratio[0] + ds_h.zoom_ratio[1]) / 2

    def make_prompt_t(bx0, bx1, by0, by1):
        pm = ds_h.create_plateau_heatmap([bx0, by0, bx1, by1], orig_h, orig_w)
        pm = cv2.resize(pm, (IMG_SIZE, IMG_SIZE))
        return torch.from_numpy(pm).unsqueeze(0).unsqueeze(0)

    # Prompt đúng vs prompt hoàn toàn lệch ra ngoài u
    prompt_good = make_prompt_t(x_min-gt_w*r, x_max+gt_w*r, y_min-gt_h*r, y_max+gt_h*r)
    # Prompt lệch: đặt vào góc đối diện
    fx = orig_w - gt_w - 20 if x_min < orig_w//2 else 20
    fy = orig_h - gt_h - 20 if y_min < orig_h//2 else 20
    prompt_bad  = make_prompt_t(fx, fx+gt_w, fy, fy+gt_h)

    image_r = cv2.resize(image_cv, (IMG_SIZE, IMG_SIZE))
    mask_r  = (cv2.resize(mask_cv, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 127)
    img_t   = torch.from_numpy((image_r.astype(np.float32)/255.0-0.5)/0.5).unsqueeze(0).unsqueeze(0)
    img_np  = image_r / 255.0
    gt_np   = mask_r.astype(np.float32)

    res_good = predict_with_check(model, img_t, prompt_good)
    res_bad  = predict_with_check(model, img_t, prompt_bad)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f"Cải tiến inference – phát hiện prompt sai | {img_name}",
                 fontsize=14, fontweight='bold')

    for row, (res, label) in enumerate([(res_good, "Prompt ĐÚNG"), (res_bad, "Prompt SAI")]):
        pm_np = (prompt_good if row == 0 else prompt_bad)[0, 0].numpy()
        pm_r  = cv2.resize(pm_np, (IMG_SIZE, IMG_SIZE)) if pm_np.shape != (IMG_SIZE, IMG_SIZE) else pm_np

        color = 'green' if not res['is_suspicious'] else 'red'
        status = "✓ Tin cậy" if not res['is_suspicious'] else "⚠ Nghi ngờ sai"

        axes[row][0].imshow(img_np, cmap='gray')
        axes[row][0].set_ylabel(f"{label}\n{status}", fontsize=10, color=color,
                                fontweight='bold', rotation=0, labelpad=70, va='center')

        axes[row][1].imshow(img_np, cmap='gray')
        axes[row][1].imshow(np.ma.masked_where(pm_r < 0.05, pm_r), cmap='magma', alpha=0.6)
        axes[row][1].set_title(f"Prompt\nConf: {res['confidence']:.3f}", fontsize=9)

        axes[row][2].imshow(img_np, cmap='gray')
        axes[row][2].imshow(np.ma.masked_where(gt_np == 0, gt_np), cmap='Greens', alpha=0.5)
        axes[row][2].set_title("Ground Truth", fontsize=9)

        axes[row][3].imshow(img_np, cmap='gray')
        axes[row][3].imshow(np.ma.masked_where(res['mask'] == 0, res['mask']), cmap='Reds', alpha=0.5)
        axes[row][3].set_title(f"Dự đoán\nDist tâm: {res['center_dist']:.1f}px", fontsize=9)

        if res['saliency'] is not None:
            axes[row][4].imshow(img_np, cmap='gray')
            axes[row][4].imshow(res['saliency'], cmap='hot', alpha=0.6)
            axes[row][4].set_title("GradCAM\n(gợi ý vùng u)", fontsize=9, color='orange')
        else:
            axes[row][4].imshow(res['prob_map'], cmap='RdYlGn')
            axes[row][4].set_title("Probability map", fontsize=9)

        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"result_inference_check_{os.path.splitext(img_name)[0]}.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Prompt đúng  → suspicious={res_good['is_suspicious']}, conf={res_good['confidence']:.3f}")
    print(f"Prompt sai   → suspicious={res_bad['is_suspicious']},  conf={res_bad['confidence']:.3f}")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_model = evaluate()
    visualize_3_scenarios(trained_model)
    visualize_inference_check(trained_model)