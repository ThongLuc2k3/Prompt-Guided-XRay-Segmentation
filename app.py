# @title
"""
app.py – Demo Tương Tác PGA-UNet (Tích hợp Cơ chế Bảo hộ Tự động)
=============================================================================
Kịch bản:
  - Bác sĩ upload ảnh, click 2 điểm tạo box (prompt).
  - TH1 (Chuẩn): Prompt rơi vào vùng xương -> Phân đoạn bình thường (Mask Đỏ).
  - TH2 (Sai): Prompt rơi vào >70% nền đen -> Bật Bảo Hộ. 
    Hệ thống âm thầm chạy GradCAM và IPR 3 vòng, trả về Mask Cứu Hộ (Mask Xanh) 
    và cảnh báo bác sĩ.
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
# 1. CẤU HÌNH HỆ THỐNG
# =========================================================
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE             = 512
USE_ENCODER_PROMPT   = True    
CONFIDENCE_THRESHOLD = 0.25   
MIN_PRED_PIXELS      = 50     

# --- THÔNG SỐ BẢO VỆ CHÍNH ---
DARK_PIXEL_THRESHOLD = -0.80  # Ngưỡng cường độ "nền đen" trên ảnh [-1, 1]
DARK_RATIO_LIMIT     = 0.70   # >70% vùng đen -> Bật bảo hộ
NUM_IPR_STEPS        = 3      # Số vòng lặp nắn nét IPR

# =========================================================
# 2. TẢI MÔ HÌNH
# =========================================================
print(f"[*] Thiết bị: {DEVICE}")

model_prompt = PGA_UNet(in_channels=1, n_classes=1, use_encoder_prompt=USE_ENCODER_PROMPT).to(DEVICE)
try:
    model_prompt.load_state_dict(
        torch.load("checkpoints/pga_unet_expB_best.pth",
                   map_location=DEVICE, weights_only=True)
    )
    model_prompt.eval()
    print("[+] PGA-UNet: Tải trọng số thành công!")
except Exception as e:
    print(f"[-] Lỗi tải mô hình: {e}")

# =========================================================
# 3. CÁC HÀM XỬ LÝ (HELPERS)
# =========================================================
def extract_lcc(binary_map: np.ndarray):
    """Lọc lấy vùng liên thông lớn nhất (bỏ nhiễu vụn)"""
    if binary_map.sum() == 0: return binary_map
    mask_uint8 = binary_map.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1: return binary_map
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.float32)

def get_centroid(binary_map: np.ndarray):
    if binary_map.sum() == 0: return None, None
    ys, xs = np.where(binary_map > 0.5)
    return float(xs.mean()), float(ys.mean())

def create_plateau_heatmap(bbox, orig_h, orig_w):
    heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
    x_min, y_min, x_max, y_max = bbox
    pad = 5
    x_min, y_min = max(0, int(x_min) - pad), max(0, int(y_min) - pad)
    x_max, y_max = min(orig_w, int(x_max) + pad), min(orig_h, int(y_max) + pad)
    heatmap[y_min:y_max, x_min:x_max] = 1.0
    return cv2.GaussianBlur(heatmap, (31, 31), 0)

def compute_gradcam(model, img_tensor):
    gradients, activations = [], []
    def fwd_hook(module, inp, out):
        activations.append(out)
        out.register_hook(lambda g: gradients.append(g))
    
    hook = model.center.register_forward_hook(fwd_hook)
    model.eval()
    img_t = img_tensor.clone().detach().to(DEVICE)
    zero_prompt = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
    try:
        out = model(img_t, zero_prompt)
        model.zero_grad()
        out.sum().backward()
    finally:
        hook.remove()
        
    if not gradients: return None
    w = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * activations[0]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# =========================================================
# 4. TRỰC QUAN HÓA (CHỈ MASK & BOX, KHÔNG HEATMAP)
# =========================================================
def overlay_clean_result(image_rgb, pred_mask, original_bbox, final_bbox=None, is_rescued=False):
    """Hàm này chỉ in ra ảnh gốc + Mask + Box (Rất sạch sẽ)"""
    result = image_rgb.copy()
    colored = np.zeros_like(result)
    
    # 1. Vẽ Mask
    if pred_mask is not None and pred_mask.max() > 0:
        if is_rescued:
            colored[pred_mask > 0] = [50, 220, 50] # Xanh lá (Mask Cứu Hộ)
        else:
            colored[pred_mask > 0] = [220, 50, 50] # Đỏ (Mask Chuẩn)
        result = cv2.addWeighted(result, 1.0, colored, 0.5, 0)
        
    # 2. Vẽ Box Cứu hộ (Màu vàng đứt nét)
    if is_rescued and final_bbox is not None:
        xb_min, yb_min, xb_max, yb_max = (int(v) for v in final_bbox)
        cv2.rectangle(result, (xb_min, yb_min), (xb_max, yb_max), (255, 215, 0), 2)
        
    # 3. Vẽ Box gốc của bác sĩ (Màu xám mờ để so sánh)
    if original_bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in original_bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), (180, 180, 180), 1)
        
    return result

# =========================================================
# 5. LÕI INFERENCE (XỬ LÝ SỰ KIỆN NÚT "DỰ ĐOÁN")
# =========================================================
def run_inference(image, points_state):
    if image is None: return None, "❌ Vui lòng tải ảnh lên trước!"
    if len(points_state) < 2: return None, "⚠️ Hãy click 2 điểm trên ảnh để tạo vùng Prompt."

    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Chuẩn bị Tensor
    img_r = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy((img_r.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).to(DEVICE)
    img_np = img_t[0,0].cpu().numpy()

    # Lấy tọa độ Prompt từ người dùng
    x1, y1 = int(points_state[0][0]), int(points_state[0][1])
    x2, y2 = int(points_state[1][0]), int(points_state[1][1])
    original_bbox = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
    
    # Tạo heatmap Prompt
    heatmap = create_plateau_heatmap(original_bbox, orig_h, orig_w)
    heatmap_r = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_t = torch.from_numpy(heatmap_r).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    # ─── BƯỚC 1: QUÉT KIỂM TRA TỶ LỆ NỀN ĐEN CỦA PROMPT ───
    pm_mask = heatmap_r > 0.3
    if pm_mask.sum() > 0:
        dark_ratio = (img_np[pm_mask] < DARK_PIXEL_THRESHOLD).sum() / pm_mask.sum()
        is_dark_bg = dark_ratio > DARK_RATIO_LIMIT
    else:
        is_dark_bg = True
        dark_ratio = 1.0

    # ─── BƯỚC 2: CHẠY SUY LUẬN BÌNH THƯỜNG TRƯỚC ───
    with torch.no_grad():
        out  = model_prompt(img_t, heatmap_t)
        prob = torch.sigmoid(out)
        pred = (prob > 0.5).float()

    confidence = prob.max().item()
    pred_np    = pred.squeeze().cpu().numpy()
    pred_area  = int(pred_np.sum())
    
    is_empty    = pred_area < MIN_PRED_PIXELS
    is_low_conf = confidence < CONFIDENCE_THRESHOLD

    # ─── BƯỚC 3: XỬ LÝ THEO KỊCH BẢN ───
    # KỊCH BẢN A: PHẢI KÍCH HOẠT BẢO HỘ (Nền đen >70% hoặc mask rỗng)
    if is_empty or is_low_conf or is_dark_bg:
        cam = compute_gradcam(model_prompt, img_t)
        
        if cam is None: # Cứu hộ thất bại
            return image_rgb, "❌ Lỗi: Prompt sai và hệ thống GradCAM không khả dụng."

        # >> Bắt đầu IPR 3 Vòng ở Backend
        py_curr, px_curr = np.unravel_index(cam.argmax(), cam.shape)
        bw, bh = 80, 80 # Kích thước box mồi
        
        ipr_mask_final = None
        ipr_box_final = None

        for v in range(1, NUM_IPR_STEPS + 1):
            pm_ipr = create_plateau_heatmap([px_curr-bw/2, py_curr-bh/2, px_curr+bw/2, py_curr+bh/2], IMG_SIZE, IMG_SIZE)
            pm_ipr_t = torch.from_numpy(pm_ipr).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                prob_ipr = torch.sigmoid(model_prompt(img_t, pm_ipr_t))[0,0].cpu().numpy()
                mask_ipr = extract_lcc((prob_ipr > 0.5).astype(np.float32))
                cx, cy = get_centroid(mask_ipr)
                if cx: 
                    px_curr, py_curr = cx, cy
                    ipr_mask_final = mask_ipr
                    ipr_box_final = [px_curr-bw/2, py_curr-bh/2, px_curr+bw/2, py_curr+bh/2]

        # Resize lại về kích thước ảnh gốc
        if ipr_mask_final is not None:
            ipr_mask_orig = cv2.resize(ipr_mask_final, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            sx, sy = orig_w / IMG_SIZE, orig_h / IMG_SIZE
            ipr_box_orig = [ipr_box_final[0]*sx, ipr_box_final[1]*sy, ipr_box_final[2]*sx, ipr_box_final[3]*sy]
        else:
            ipr_mask_orig, ipr_box_orig = None, None

        # Trả về kết quả sạch (Không có Heatmap)
        result = overlay_clean_result(image_rgb, ipr_mask_orig, original_bbox, ipr_box_orig, is_rescued=True)
        
        reasons = []
        if is_dark_bg: reasons.append(f"Chứa {dark_ratio*100:.1f}% nền đen")
        elif is_empty: reasons.append("Không tìm thấy u trong vùng khoanh")
        
        msg = (f"🛡️ KÍCH HOẠT BẢO HỘ TỰ ĐỘNG!\n"
               f"- Phát hiện Prompt sai: {' / '.join(reasons)}.\n"
               f"- Hệ thống đã bỏ qua Prompt của bạn, tự động dò tìm bằng GradCAM và nắn nét IPR {NUM_IPR_STEPS} vòng.\n"
               f"-> Kết quả cứu hộ: Khối u được tìm thấy ở vị trí mới (Mask Màu Xanh).")
        return result, msg

    # KỊCH BẢN B: PROMPT CHUẨN (Suy luận bình thường)
    pred_orig = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    result = overlay_clean_result(image_rgb, pred_orig, original_bbox, is_rescued=False)
    
    area_pct = pred_np.sum() / (IMG_SIZE * IMG_SIZE) * 100
    msg = (f"✅ PHÂN ĐOẠN CHUẨN MỰC!\n"
           f"- Mô hình tự tin với Prompt này (Độ tin cậy: {confidence:.3f}).\n"
           f"- Không cần bật bảo hộ. Mask xuất ra có Màu Đỏ (Diện tích: {area_pct:.1f}%).")
    return result, msg

# =========================================================
# 6. GIAO DIỆN APP TƯƠNG TÁC (GRADIO)
# =========================================================
def get_clicks(img, evt: gr.SelectData, points_state):
    """Hàm xử lý logic click chuột lấy tọa độ box"""
    if img is None: return img, points_state, "Vui lòng tải ảnh lên!"
    
    pts = list(points_state)
    if len(pts) >= 2: pts = [] # Reset nếu click điểm thứ 3
    
    pts.append((int(evt.index[0]), int(evt.index[1])))
    img_drawn = img.copy()

    for p in pts:
        cv2.circle(img_drawn, p, 6, (255, 60, 60), -1)
        cv2.circle(img_drawn, p, 7, (255, 255, 255), 1)

    if len(pts) == 2:
        x1, y1 = pts[0]; x2, y2 = pts[1]
        cv2.rectangle(img_drawn, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), (50, 220, 50), 2)
        return img_drawn, pts, "🎯 Đã vẽ xong Box! Hãy nhấn nút Dự Đoán."

    return img_drawn, pts, f"Đã lấy góc thứ nhất: {pts[-1]} | Hãy click góc đối diện."

def reset_all():
    return None, None, [], "🔄 Đã Reset. Hãy tải ảnh mới và click lại."

# --- Cấu hình CSS cho đẹp ---
_CSS = """
.status-box textarea { font-size: 14px !important; line-height: 1.6 !important; font-weight: bold; }
.legend-row { display: flex; gap: 20px; font-size: 14px; margin-top: 10px; flex-wrap: wrap; justify-content: center;}
.legend-item { display: flex; align-items: center; gap: 8px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=_CSS, title="Demo Cứu Hộ PGA-UNet") as demo:
    gr.Markdown("<h2 style='text-align: center;'>🦴 Demo Hệ Thống Phân Đoạn X-quang Có Cơ Chế Bảo Hộ Thông Minh</h2>")
    gr.Markdown("**Hướng dẫn:** Tải ảnh lên -> Dùng chuột click 2 điểm (tạo box) -> Nhấn Dự Đoán. Hãy thử cố tình khoanh ra ngoài không khí (vùng đen) để xem AI tự sửa sai nhé!")

    points_state = gr.State([])

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Ảnh Đầu Vào (Click tạo Prompt)", height=480)
            status_box = gr.Textbox(label="Báo cáo Hệ thống", value="Đang chờ tải ảnh...", interactive=False, lines=4, elem_classes=["status-box"])
            with gr.Row():
                btn_predict = gr.Button("🧠 DỰ ĐOÁN", variant="primary")
                btn_reset   = gr.Button("🔄 Xóa / Reset", variant="secondary")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Kết Quả Phân Đoạn", height=480)
            gr.HTML("""
            <div class="legend-row">
              <div class="legend-item"><span style="display:inline-block;width:16px;height:16px;background:rgba(220,50,50,0.8);border-radius:3px"></span>Mask Bình Thường</div>
              <div class="legend-item"><span style="display:inline-block;width:16px;height:16px;background:rgba(50,220,50,0.8);border-radius:3px"></span>Mask Cứu Hộ (AI tự sửa)</div>
              <div class="legend-item"><span style="display:inline-block;width:16px;height:16px;border:2px solid #aaa"></span>Box Bác Sĩ (Xám)</div>
              <div class="legend-item"><span style="display:inline-block;width:16px;height:16px;border:2px solid #FFD700"></span>Box IPR nắn lại (Vàng)</div>
            </div>
            """)

    # Events
    input_image.select(fn=get_clicks, inputs=[input_image, points_state], outputs=[input_image, points_state, status_box])
    btn_predict.click(fn=run_inference, inputs=[input_image, points_state], outputs=[output_image, status_box])
    btn_reset.click(fn=reset_all, outputs=[input_image, output_image, points_state, status_box])

if __name__ == "__main__":
    demo.launch(share=True, debug=False)