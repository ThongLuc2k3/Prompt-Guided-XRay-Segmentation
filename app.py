"""
app.py – Demo tương tác PGA-UNet
=================================
Cách dùng:
  1. Tải ảnh X-quang lên
  2. Click 2 điểm (góc trên-trái và góc dưới-phải) để vẽ box vùng nghi ngờ
  3. Nhấn "Dự Đoán"
 
Kết quả:
  - Vùng đỏ   = mask khối u (khi prompt đúng, confidence cao)
  - ★ + viền vàng = GradCAM gợi ý vị trí u thật sự (khi prompt sai / không tìm thấy)
  - Viền xám  = vị trí box prompt gốc của người dùng
 
pip install gradio torch torchvision opencv-python
"""
 
import warnings
warnings.filterwarnings("ignore")
 
import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import cv2
 
from models.networks.prompt_unet_2D import PGA_UNet
 
# =========================================================
# CẤU HÌNH
# =========================================================
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE             = 512
USE_ENCODER_PROMPT   = True    # Đồng bộ với lúc train
CONFIDENCE_THRESHOLD = 0.25   # Dưới ngưỡng này → kích hoạt GradCAM
MIN_PRED_PIXELS      = 50     # Ít hơn số pixel này → coi là rỗng
CAM_BOX_THRESHOLD    = 0.45   # Ngưỡng CAM để xác định bounding box gợi ý
 
# =========================================================
# TẢI MÔ HÌNH
# =========================================================
print(f"[*] Thiết bị: {DEVICE}")
 
# Baseline (tuỳ chọn)
model_unet = None
try:
    from models.networks.unet_2D import unet_2D
    model_unet = unet_2D(in_channels=1, n_classes=1).to(DEVICE)
    model_unet.load_state_dict(
        torch.load("checkpoints_Unet2D/Unet_2D/Unet2D_best.pth",
                   map_location=DEVICE, weights_only=True)
    )
    model_unet.eval()
    print("[+] U-Net baseline: OK")
except Exception as e:
    print(f"[-] U-Net baseline không tải được: {e}")
 
# PGA-UNet (model chính)
model_prompt = PGA_UNet(
    in_channels=1, n_classes=1,
    use_encoder_prompt=USE_ENCODER_PROMPT
).to(DEVICE)
try:
    model_prompt.load_state_dict(
        torch.load("checkpoints/Prompt_ATT_Unet2D/pga_unet_expB_best.pth",
                   map_location=DEVICE, weights_only=True)
    )
    model_prompt.eval()
    print("[+] PGA-UNet: OK")
except Exception as e:
    print(f"[-] PGA-UNet không tải được: {e}")
 
# =========================================================
# HELPER: TẠO HEATMAP PROMPT
# =========================================================
def create_plateau_heatmap(bbox, orig_h, orig_w):
    """Tạo Gaussian plateau heatmap từ bounding box – đồng nhất với lúc train."""
    heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
    x_min, y_min, x_max, y_max = bbox
    pad = 5
    x_min = max(0,       int(x_min) - pad)
    y_min = max(0,       int(y_min) - pad)
    x_max = min(orig_w,  int(x_max) + pad)
    y_max = min(orig_h,  int(y_max) + pad)
    heatmap[y_min:y_max, x_min:x_max] = 1.0
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    return heatmap
 
# =========================================================
# HELPER: GRADCAM TỪ BOTTLENECK (zero_prompt)
# =========================================================
def compute_gradcam(model, img_tensor):
    """
    GradCAM từ model.center (bottleneck) với zero_prompt.
    Tìm vùng model "tự thấy" bất kể prompt của người dùng ở đâu.
    Trả về numpy (H, W) ∈ [0,1] hoặc None nếu thất bại.
    """
    gradients:  list = []
    activations: list = []
 
    def fwd_hook(module, inp, out):
        activations.append(out)
        out.register_hook(lambda g: gradients.append(g))
 
    hook = model.center.register_forward_hook(fwd_hook)
    model.eval()
 
    img_t       = img_tensor.clone().detach().to(DEVICE)
    zero_prompt = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
 
    try:
        out = model(img_t, zero_prompt)
        model.zero_grad()
        out.sum().backward()
    finally:
        hook.remove()
 
    if not gradients or not activations:
        return None
 
    g   = gradients[0]   # (1, C, h, w)
    a   = activations[0] # (1, C, h, w)
    w   = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                        mode='bilinear', align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    vmin, vmax = cam.min(), cam.max()
    return (cam - vmin) / (vmax - vmin + 1e-8)
 
# =========================================================
# HELPER: PHỦ MASK ĐỎ (kết quả bình thường)
# =========================================================
def overlay_mask(image_rgb, pred_mask, prompt_bbox=None):
    """Phủ vùng đỏ cho mask dự đoán và viền vàng cho prompt box."""
    result = image_rgb.copy()
    colored = np.zeros_like(result)
    colored[pred_mask > 0] = [220, 50, 50]
    result = cv2.addWeighted(result, 1.0, colored, 0.5, 0)
    if prompt_bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in prompt_bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 220, 0), 2)
    return result
 
# =========================================================
# HELPER: PHỦ GRADCAM + ★ + BOX GỢI Ý
# =========================================================
def overlay_gradcam_suggestion(image_rgb, cam, orig_w, orig_h, prompt_bbox=None):
    """
    Khi dự đoán rỗng / kém tin cậy, hiển thị:
      • Heatmap jet làm nền gợi ý
      • Bounding box mờ vàng quanh vùng activation cao nhất
      • Ngôi sao ★ tại đỉnh activation (vị trí gợi ý tâm u)
      • Viền xám mờ cho prompt gốc của người dùng
    """
    result = image_rgb.copy()
 
    # ── 1. Resize CAM về kích thước ảnh gốc ──────────────
    cam_orig = cv2.resize(cam, (orig_w, orig_h))
 
    # ── 2. Heatmap jet overlay (chỉ vùng nóng > 0.25) ────
    heatmap_u8  = (cam_orig * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
 
    hot = cam_orig > 0.25
    result_f = result.astype(np.float32)
    result_f[hot] = result_f[hot] * 0.50 + heatmap_rgb[hot] * 0.50
    result = np.clip(result_f, 0, 255).astype(np.uint8)
 
    # ── 3. Tìm vùng bounding box từ CAM ──────────────────
    cam_thresh = (cam_orig > CAM_BOX_THRESHOLD).astype(np.uint8)
 
    if cam_thresh.any():
        # Giữ largest connected component để tránh nhiều box lẻ
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cam_thresh, connectivity=8
        )
        if n_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cam_thresh = (labels == largest).astype(np.uint8)
 
        ys_c, xs_c = np.where(cam_thresh)
        if len(ys_c) > 0:
            xb_min, xb_max = int(xs_c.min()), int(xs_c.max())
            yb_min, yb_max = int(ys_c.min()), int(ys_c.max())
 
            # Semi-transparent yellow fill
            box_fill = result.copy()
            cv2.rectangle(box_fill,
                          (xb_min, yb_min), (xb_max, yb_max),
                          (255, 215, 0), cv2.FILLED)
            result = cv2.addWeighted(result, 0.80, box_fill, 0.20, 0)
 
            # Viền liền vàng đậm
            cv2.rectangle(result,
                          (xb_min, yb_min), (xb_max, yb_max),
                          (255, 215, 0), 2)
 
            # Viền nét đứt (dash) bên trong để rõ hơn
            dash = 10
            for x in range(xb_min, xb_max, dash * 2):
                x2d = min(x + dash, xb_max)
                cv2.line(result, (x, yb_min), (x2d, yb_min), (255, 255, 120), 1)
                cv2.line(result, (x, yb_max), (x2d, yb_max), (255, 255, 120), 1)
            for y in range(yb_min, yb_max, dash * 2):
                y2d = min(y + dash, yb_max)
                cv2.line(result, (xb_min, y), (xb_min, y2d), (255, 255, 120), 1)
                cv2.line(result, (xb_max, y), (xb_max, y2d), (255, 255, 120), 1)
 
    # ── 4. Ngôi sao ★ tại đỉnh CAM ───────────────────────
    peak_y, peak_x = np.unravel_index(cam_orig.argmax(), cam_orig.shape)
    px, py = int(peak_x), int(peak_y)
 
    cv2.drawMarker(result, (px, py),
                   (255, 255, 0), cv2.MARKER_STAR, 32, 3, cv2.LINE_AA)
    # Chấm trắng ở tâm ngôi sao để nổi bật trên nền tối/sáng
    cv2.circle(result, (px, py), 4, (255, 255, 255), -1)
    cv2.circle(result, (px, py), 4, (180, 140, 0),   1)
 
    # ── 5. Viền xám mờ cho box prompt gốc ────────────────
    if prompt_bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in prompt_bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), (180, 180, 180), 1)
 
    return result
 
# =========================================================
# SỰ KIỆN: CLICK CHUỘT LẤY 2 ĐIỂM
# =========================================================
def get_clicks(img, evt: gr.SelectData, points_state):
    if img is None:
        return img, points_state, "Vui lòng tải ảnh lên trước!"
 
    pts = list(points_state)
 
    # Sau 2 điểm, click tiếp = reset và bắt đầu lại
    if len(pts) >= 2:
        pts = []
 
    pts.append((int(evt.index[0]), int(evt.index[1])))
    img_drawn = img.copy()
 
    # Vẽ điểm đã chấm
    for p in pts:
        cv2.circle(img_drawn, p, 6,  (255,  60, 60), -1)
        cv2.circle(img_drawn, p, 7,  (255, 255, 255),  1)
 
    if len(pts) == 2:
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        cv2.rectangle(img_drawn,
                      (min(x1,x2), min(y1,y2)),
                      (max(x1,x2), max(y1,y2)),
                      (50, 220, 50), 2)
        return img_drawn, pts, "✅ Đã vẽ box! Nhấn Dự Đoán."
 
    return img_drawn, pts, f"Đã lấy điểm 1: {pts[-1]}  |  Click điểm 2 (góc đối diện)."
 
# =========================================================
# SỰ KIỆN: DỰ ĐOÁN
# =========================================================
def run_inference(image, model_choice, points_state):
    if image is None:
        return None, "❌ Chưa có ảnh. Hãy tải ảnh lên."
 
    orig_h, orig_w = image.shape[:2]
 
    # Đảm bảo ảnh RGB để overlay màu
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
 
    # Chuyển sang grayscale để feed model
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
 
    # Preprocess: resize + Z-score (đồng nhất với dataset.py)
    img_r = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy(
        (img_r.astype(np.float32) / 255.0 - 0.5) / 0.5
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)
 
    # ── U-Net Baseline (không cần prompt) ─────────────────
    if model_choice == "U-Net 2D (Baseline)":
        if model_unet is None:
            return None, "❌ Chưa tải được U-Net baseline."
        with torch.no_grad():
            out  = model_unet(img_t)
            pred = (torch.sigmoid(out) > 0.5).float().squeeze().cpu().numpy()
        pred_orig = cv2.resize(pred, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
        result = overlay_mask(image_rgb, pred_orig.astype(np.uint8))
        area   = pred_orig.sum() / (orig_h * orig_w) * 100
        return result, f"✅ U-Net Baseline (không dùng prompt)\nDiện tích dự đoán: {area:.1f}%"
 
    # ── PGA-UNet với Prompt ───────────────────────────────
    if len(points_state) < 2:
        return None, "⚠️ Chưa đủ 2 điểm!\nHãy click 2 điểm lên ảnh để vẽ bounding box."
 
    x1, y1 = int(points_state[0][0]), int(points_state[0][1])
    x2, y2 = int(points_state[1][0]), int(points_state[1][1])
    prompt_bbox = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
 
    # Tạo prompt heatmap
    heatmap   = create_plateau_heatmap(prompt_bbox, orig_h, orig_w)
    heatmap_r = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_t = torch.from_numpy(heatmap_r).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
 
    # Forward pass
    with torch.no_grad():
        out  = model_prompt(img_t, heatmap_t)
        prob = torch.sigmoid(out)
        pred = (prob > 0.5).float()
 
    confidence = prob.max().item()
    pred_np    = pred.squeeze().cpu().numpy()
    pred_area  = int(pred_np.sum())
 
    is_empty    = pred_area < MIN_PRED_PIXELS
    is_low_conf = confidence < CONFIDENCE_THRESHOLD
 
    # ── Dự đoán rỗng hoặc kém tin cậy → GradCAM ─────────
    if is_empty or is_low_conf:
        cam = compute_gradcam(model_prompt, img_t)
 
        if cam is None:
            # Fallback: show prob map nếu GradCAM thất bại
            prob_np   = prob.squeeze().cpu().numpy()
            prob_orig = cv2.resize(prob_np, (orig_w, orig_h))
            hm_u8     = (prob_orig * 255).astype(np.uint8)
            hm_rgb    = cv2.cvtColor(
                cv2.applyColorMap(hm_u8, cv2.COLORMAP_HOT),
                cv2.COLOR_BGR2RGB
            )
            result_f  = image_rgb.astype(np.float32)
            hot_mask  = prob_orig > 0.05
            result_f[hot_mask] = (result_f[hot_mask] * 0.55 +
                                  hm_rgb[hot_mask].astype(np.float32) * 0.45)
            result = np.clip(result_f, 0, 255).astype(np.uint8)
            if prompt_bbox:
                cv2.rectangle(result,
                              (prompt_bbox[0], prompt_bbox[1]),
                              (prompt_bbox[2], prompt_bbox[3]),
                              (180, 180, 180), 1)
            reason = "không phát hiện khối u" if is_empty \
                     else f"confidence rất thấp ({confidence:.3f})"
            return result, (f"⚠️ Prompt {reason}.\n"
                            f"Hiển thị probability map (GradCAM không khả dụng).")
 
        # Overlay GradCAM với box gợi ý và ngôi sao
        result = overlay_gradcam_suggestion(
            image_rgb, cam, orig_w, orig_h, prompt_bbox
        )
 
        if is_empty:
            reason_vi = f"không tìm thấy khối u trong vùng box ({pred_area} px)"
        else:
            reason_vi = f"độ tin cậy thấp ({confidence:.3f} < {CONFIDENCE_THRESHOLD})"
 
        return result, (
            f"⚠️  {reason_vi}\n"
            f"GradCAM gợi ý vùng khả nghi (★ = tâm activation cao nhất).\n"
            f"Thử chấm lại vào vùng được đánh dấu màu vàng."
        )
 
    # ── Dự đoán hợp lệ → mask đỏ ─────────────────────────
    pred_orig = cv2.resize(pred_np, (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST)
    result    = overlay_mask(image_rgb, pred_orig.astype(np.uint8), prompt_bbox)
    area_pct  = pred_np.sum() / (IMG_SIZE * IMG_SIZE) * 100
 
    return result, (
        f"✅ Phân đoạn thành công!\n"
        f"Confidence: {confidence:.3f}  |  Diện tích: {area_pct:.1f}%"
    )
 
# =========================================================
# RESET
# =========================================================
def reset_all():
    return None, None, [], "Đã reset. Tải ảnh và click 2 điểm để bắt đầu."
 
# =========================================================
# GIAO DIỆN GRADIO
# =========================================================
_CSS = """
.status-box textarea { font-size: 13px !important; line-height: 1.6 !important; }
.legend-row { display: flex; gap: 18px; font-size: 13px; margin-top: 6px; }
.legend-item { display: flex; align-items: center; gap: 5px; }
"""
 
with gr.Blocks(theme=gr.themes.Soft(), css=_CSS,
               title="PGA-UNet – X-ray Segmentation") as demo:
 
    gr.Markdown("## 🦴 Phân đoạn tổn thương xương – Prompt-Guided Attention U-Net")
    gr.Markdown(
        "**Cách dùng:** Tải ảnh X-quang → Click 2 điểm (góc trên-trái & dưới-phải "
        "của vùng nghi ngờ) → **Dự Đoán**"
    )
 
    points_state = gr.State([])
 
    with gr.Row(equal_height=True):
        # ── Cột trái: input ───────────────────────────────
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="numpy",
                label="Ảnh X-quang gốc  (click để chấm điểm)",
                height=480,
            )
 
            model_radio = gr.Radio(
                choices=["PGA-UNet (Prompt)", "U-Net 2D (Baseline)"],
                value="PGA-UNet (Prompt)",
                label="Mô hình"
            )
 
            status_box = gr.Textbox(
                label="Trạng thái",
                value="Đang chờ... Tải ảnh và click 2 điểm.",
                interactive=False,
                lines=3,
                elem_classes=["status-box"],
            )
 
            with gr.Row():
                btn_predict = gr.Button("🧠  Dự Đoán", variant="primary",  scale=3)
                btn_reset   = gr.Button("🗑️  Reset",    variant="secondary", scale=1)
 
        # ── Cột phải: output ──────────────────────────────
        with gr.Column(scale=1):
            output_image = gr.Image(
                type="numpy",
                label="Kết quả phân đoạn",
                height=480,
            )
            gr.HTML("""
            <div class="legend-row">
              <div class="legend-item">
                <span style="display:inline-block;width:14px;height:14px;
                      background:rgba(220,50,50,0.7);border-radius:2px"></span>
                Mask khối u (prompt đúng)
              </div>
              <div class="legend-item">
                <span style="font-size:16px;line-height:1">⭐</span>
                GradCAM gợi ý (prompt sai / rỗng)
              </div>
              <div class="legend-item">
                <span style="display:inline-block;width:14px;height:14px;
                      border:1px solid #aaa"></span>
                Box prompt gốc
              </div>
            </div>
            """)
 
    # ── Events ────────────────────────────────────────────
    input_image.select(
        fn=get_clicks,
        inputs=[input_image, points_state],
        outputs=[input_image, points_state, status_box],
    )
 
    btn_predict.click(
        fn=run_inference,
        inputs=[input_image, model_radio, points_state],
        outputs=[output_image, status_box],
    )
 
    btn_reset.click(
        fn=reset_all,
        outputs=[input_image, output_image, points_state, status_box],
    )
 
if __name__ == "__main__":
    demo.launch(share=True, debug=False)