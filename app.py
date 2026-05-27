# @title
import warnings
warnings.filterwarnings("ignore")

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import timm
from PIL import Image

# Nhập mô hình phân đoạn
from models.networks.prompt_unet_2D import PGA_UNet

# =========================================================
# 1. CẤU HÌNH & TẢI MÔ HÌNH CHUNG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[*] HỆ THỐNG KHỞI CHẠY TRÊN THIẾT BỊ: {DEVICE}")

# --- Tải Mô Hình Sàng Lọc (MobileNetV4) ---
try:
    model_class = timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r384_in1k', pretrained=False)
    in_features = model_class.classifier.in_features
    model_class.classifier = nn.Linear(in_features, 1)
    model_class = model_class.to(DEVICE)
    data_config = timm.data.resolve_model_data_config(model_class)
    class_transform = timm.data.create_transform(**data_config, is_training=False)
    model_class.load_state_dict(torch.load("checkpoints/best_mobilenetv4.pth", map_location=DEVICE))
    model_class.eval()
    print("[+] SÀNG LỌC: MobileNetV4 tải thành công!")
except Exception as e:
    print(f"[-] Lỗi tải MobileNetV4: {e}")

# --- Tải Mô Hình Phân Đoạn (PGA-UNet) ---
IMG_SIZE = 512
DARK_PIXEL_THRESHOLD = -0.80
DARK_RATIO_LIMIT = 0.70
NUM_IPR_STEPS = 3

model_prompt = PGA_UNet(in_channels=1, n_classes=1, use_encoder_prompt=True).to(DEVICE)
try:
    model_prompt.load_state_dict(torch.load("checkpoints/pga_unet_expB_best.pth", map_location=DEVICE, weights_only=True))
    model_prompt.eval()
    print("[+] PHÂN ĐOẠN: PGA-UNet tải thành công!")
except Exception as e:
    print(f"[-] Lỗi tải PGA-UNet: {e}")

# =========================================================
# 2. HÀM XỬ LÝ (HELPERS)
# =========================================================
def extract_lcc(binary_map):
    if binary_map.sum() == 0: return binary_map
    mask_uint8 = binary_map.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1: return binary_map
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.float32)

def get_centroid(binary_map):
    if binary_map.sum() == 0: return None, None
    ys, xs = np.where(binary_map > 0.5)
    return float(xs.mean()), float(ys.mean())

def create_plateau_heatmap(bbox, orig_h, orig_w):
    heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
    pad = 5
    x_min, y_min = max(0, int(bbox[0])-pad), max(0, int(bbox[1])-pad)
    x_max, y_max = min(orig_w, int(bbox[2])+pad), min(orig_h, int(bbox[3])+pad)
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

def overlay_clean_result(image_rgb, pred_mask, original_bbox, final_bbox=None, is_rescued=False):
    result = image_rgb.copy()
    colored = np.zeros_like(result)
    if pred_mask is not None and pred_mask.max() > 0:
        colored[pred_mask > 0] = [50, 220, 50] if is_rescued else [220, 50, 50]
        result = cv2.addWeighted(result, 1.0, colored, 0.5, 0)
    if is_rescued and final_bbox is not None:
        cv2.rectangle(result, (int(final_bbox[0]), int(final_bbox[1])), (int(final_bbox[2]), int(final_bbox[3])), (255, 215, 0), 2)
    if original_bbox is not None:
        cv2.rectangle(result, (int(original_bbox[0]), int(original_bbox[1])), (int(original_bbox[2]), int(original_bbox[3])), (180, 180, 180), 1)
    return result

# =========================================================
# 3. LOGIC XỬ LÝ SỰ KIỆN GIAO DIỆN
# =========================================================

# --- Logic Bước 1: Sàng Lọc ---
def run_classification(image):
    if image is None: return "❌ Vui lòng tải ảnh lên trước!"
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
    pil_img = Image.fromarray(image_rgb)
    input_tensor = class_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model_class(input_tensor)
        prob = torch.sigmoid(output).item()
        pred_class = 1 if prob > 0.5 else 0
    ai_result = "CÓ BỆNH" if pred_class == 1 else "KHÔNG BỆNH"
    return f"🤖 AI Dự đoán: {ai_result} (Độ tin cậy: {prob if pred_class == 1 else 1 - prob:.3f})"

def confirm_nosick():
    html = """<div style='background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb; font-size: 16px; font-weight: bold;'>
    ✅ Chúc mừng! Bệnh nhân khỏe mạnh, không phát hiện bệnh lý xương. Kết thúc quy trình.
    </div>"""
    return html, gr.update(visible=False)

def confirm_sick():
    html = """<div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; text-align: center; border: 1px solid #f5c6cb; font-size: 16px; font-weight: bold;'>
    ⚠️ Cảnh báo! Xác nhận CÓ BỆNH. Vui lòng chuyển đến bước định vị khối u.
    </div>"""
    return html, gr.update(visible=True)

# --- Logic Chuyển Màn Hình Siêu Tốc ---
def switch_to_segmentation(img):
    if img is None: return [gr.update()]*8
    clean_img = img.copy() # Lưu lại bản sạch để chạy AI
    return (
        gr.update(visible=False), # Ẩn Controls Sàng lọc
        gr.update(visible=False), # Ẩn Output Sàng lọc
        gr.update(visible=True),  # Hiện Controls Phân đoạn
        gr.update(visible=True),  # Hiện Output Phân đoạn
        clean_img,                # Cập nhật ảnh gốc sạch
        [],                       # Reset các điểm đã chấm
        "seg",                    # Cập nhật cờ trạng thái bước
        "Đang chờ khoanh Box..."  # Reset status box
    )

def switch_to_classification():
    return (
        gr.update(visible=True),   # Hiện Controls Sàng lọc
        gr.update(visible=True),   # Hiện Output Sàng lọc
        gr.update(visible=False),  # Ẩn Controls Phân đoạn
        gr.update(visible=False),  # Ẩn Output Phân đoạn
        None,                      # Xóa ảnh đầu vào
        None,                      # Xóa ảnh kết quả
        "cls",                     # Trở về trạng thái sàng lọc
        "Chờ dự đoán...",          # Reset Textbox
        "",                        # Xóa HTML thông báo
        gr.update(visible=False)   # Ẩn nút Chuyển tiếp
    )

# --- Logic Bước 2: Phân Đoạn ---
def get_clicks(img, evt: gr.SelectData, points_state, step):
    if step != "seg":
        return img, points_state, "Vui lòng hoàn thành bước Sàng lọc trước."
    if img is None: return img, points_state, "Vui lòng tải ảnh!"

    pts = list(points_state)
    if len(pts) >= 2: pts = []
    pts.append((int(evt.index[0]), int(evt.index[1])))
    img_drawn = img.copy()

    for p in pts:
        cv2.circle(img_drawn, p, 6, (255, 60, 60), -1)
        cv2.circle(img_drawn, p, 7, (255, 255, 255), 1)

    if len(pts) == 2:
        cv2.rectangle(img_drawn, (min(pts[0][0],pts[1][0]), min(pts[0][1],pts[1][1])), (max(pts[0][0],pts[1][0]), max(pts[0][1],pts[1][1])), (50, 220, 50), 2)
        return img_drawn, pts, "🎯 Đã vẽ xong Box! Nhấn Dự Đoán Mask."
    return img_drawn, pts, f"Đã lấy góc 1: {pts[-1]} | Click góc 2."

def reset_prompt(clean_image):
    return clean_image, [], "🔄 Đã đặt lại khung. Hãy click 2 điểm để khoanh lại."

def run_segmentation(display_image, points_state, clean_image):
    if clean_image is None: clean_image = display_image # Fallback an toàn
    if len(points_state) < 2: return None, "⚠️ Cần click 2 điểm để tạo Box."

    orig_h, orig_w = clean_image.shape[:2]
    # DÙNG ẢNH SẠCH ĐỂ CHẠY AI (Tránh AI nhìn thấy nét vẽ xanh)
    image_rgb = cv2.cvtColor(clean_image, cv2.COLOR_GRAY2RGB) if len(clean_image.shape) == 2 else clean_image.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_r = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy((img_r.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).to(DEVICE)
    img_np = img_t[0,0].cpu().numpy()

    x1, y1 = int(points_state[0][0]), int(points_state[0][1])
    x2, y2 = int(points_state[1][0]), int(points_state[1][1])
    original_bbox = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]

    heatmap = create_plateau_heatmap(original_bbox, orig_h, orig_w)
    heatmap_r = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_t = torch.from_numpy(heatmap_r).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    pm_mask = heatmap_r > 0.3
    is_dark_bg = ((img_np[pm_mask] < DARK_PIXEL_THRESHOLD).sum() / pm_mask.sum() > DARK_RATIO_LIMIT) if pm_mask.sum() > 0 else True

    with torch.no_grad():
        prob = torch.sigmoid(model_prompt(img_t, heatmap_t))
        pred = (prob > 0.5).float()

    confidence = prob.max().item()
    pred_area = int(pred.squeeze().cpu().numpy().sum())

    is_empty = pred_area < 50
    is_low_conf = confidence < 0.25

    if is_empty or is_low_conf or is_dark_bg:
        cam = compute_gradcam(model_prompt, img_t)
        if cam is None: return image_rgb, "❌ Lỗi: GradCAM không khả dụng."

        py_curr, px_curr = np.unravel_index(cam.argmax(), cam.shape)
        bw, bh = 80, 80
        ipr_mask_final, ipr_box_final = None, None

        for v in range(1, NUM_IPR_STEPS + 1):
            pm_ipr = create_plateau_heatmap([px_curr-bw/2, py_curr-bh/2, px_curr+bw/2, py_curr+bh/2], IMG_SIZE, IMG_SIZE)
            pm_ipr_t = torch.from_numpy(pm_ipr).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mask_ipr = extract_lcc((torch.sigmoid(model_prompt(img_t, pm_ipr_t))[0,0].cpu().numpy() > 0.5).astype(np.float32))
                cx, cy = get_centroid(mask_ipr)
                if cx:
                    px_curr, py_curr = cx, cy
                    ipr_mask_final = mask_ipr
                    ipr_box_final = [px_curr-bw/2, py_curr-bh/2, px_curr+bw/2, py_curr+bh/2]

        if ipr_mask_final is not None:
            ipr_mask_orig = cv2.resize(ipr_mask_final, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            sx, sy = orig_w / IMG_SIZE, orig_h / IMG_SIZE
            ipr_box_orig = [ipr_box_final[0]*sx, ipr_box_final[1]*sy, ipr_box_final[2]*sx, ipr_box_final[3]*sy]
        else:
            ipr_mask_orig, ipr_box_orig = None, None

        result = overlay_clean_result(image_rgb, ipr_mask_orig, original_bbox, ipr_box_orig, is_rescued=True)
        msg = f"🛡️ BẢO HỘ TỰ ĐỘNG! Hệ thống phát hiện khoanh lệch, đã tự nắn nét {NUM_IPR_STEPS} vòng -> Mask Xanh."
        return result, msg

    pred_orig = cv2.resize(pred.squeeze().cpu().numpy(), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    result = overlay_clean_result(image_rgb, pred_orig, original_bbox, is_rescued=False)
    return result, f"✅ PHÂN ĐOẠN CHUẨN MỰC! (Độ tin cậy: {confidence:.3f}) -> Mask Đỏ."

# =========================================================
# 4. GIAO DIỆN GRADIO LIỀN MẠCH (SPA)
# =========================================================
_CSS = """
.class-box textarea { font-size: 18px !important; text-align: center !important; font-weight: 900 !important; color: #104E8B !important;}
.next-btn { background-color: #dc3545 !important; color: white !important; font-weight: bold !important; font-size: 16px !important; margin-top: 10px;}
.legend-row { display: flex; gap: 15px; font-size: 13px; margin-top: 10px; flex-wrap: wrap; justify-content: center;}
.legend-item { display: flex; align-items: center; gap: 5px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=_CSS, title="Hệ Thống Phân Đoạn AI") as demo:

    # Biến trạng thái toàn cục
    points_state = gr.State([])
    saved_original_image = gr.State(None)
    current_step = gr.State("cls")

    gr.Markdown("<h2 style='text-align: center;'>🦴 HỆ THỐNG CHẨN ĐOÁN VÀ ĐỊNH VỊ BỆNH LÝ XƯƠNG (AI TÍCH HỢP)</h2>")

    with gr.Row():
        # --- CỘT TRÁI: Ô ẢNH CHUNG VÀ CÁC NÚT ĐIỀU KHIỂN ---
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Ảnh X-quang Bệnh Nhân", height=480)

            # Điều khiển Sàng Lọc
            with gr.Group(visible=True) as group_ctrl_cls:
                btn_predict_cls = gr.Button("🧠 1. DỰ ĐOÁN SÀNG LỌC", variant="primary")

            # Điều khiển Phân Đoạn
            with gr.Group(visible=False) as group_ctrl_seg:
                status_box_seg = gr.Textbox(label="Báo cáo Hệ thống", value="Đang chờ khoanh Box...", interactive=False)
                with gr.Row():
                    btn_predict_seg = gr.Button("🧠 2. DỰ ĐOÁN MASK", variant="primary")
                    btn_reset_seg   = gr.Button("🔄 Đặt lại khung", variant="secondary")

        # --- CỘT PHẢI: KẾT QUẢ ---
        with gr.Column():

            # Kết quả Sàng Lọc
            with gr.Group(visible=True) as group_out_cls:
                class_output = gr.Textbox(label="Kết Quả AI Sàng Lọc", value="Chờ dự đoán...", interactive=False, elem_classes=["class-box"])
                gr.Markdown("<p style='text-align: center; font-weight: bold; margin-top: 10px;'>BÁC SĨ XÁC NHẬN KẾT QUẢ</p>")
                with gr.Row():
                    btn_nosick = gr.Button("✅ KHÔNG BỆNH", variant="secondary")
                    btn_sick   = gr.Button("⚠️ CÓ BỆNH", variant="secondary")
                html_alert = gr.HTML("")
                btn_go_seg = gr.Button("➡️ CHUYỂN ĐẾN PHÂN ĐOẠN KHỐI U", visible=False, elem_classes=["next-btn"])

            # Kết quả Phân Đoạn
            with gr.Group(visible=False) as group_out_seg:
                output_image_seg = gr.Image(type="numpy", label="Bản Đồ Vị Trí Khối U", height=480)
                gr.HTML("""
                <div class="legend-row">
                  <div class="legend-item"><span style="display:inline-block;width:14px;height:14px;background:rgba(220,50,50,0.7)"></span> Mask Chuẩn (Đỏ)</div>
                  <div class="legend-item"><span style="display:inline-block;width:14px;height:14px;background:rgba(50,220,50,0.7)"></span> Mask Cứu Hộ (Xanh)</div>
                  <div class="legend-item"><span style="display:inline-block;width:14px;height:14px;border:1px solid #aaa"></span> Box Gốc</div>
                  <div class="legend-item"><span style="display:inline-block;width:14px;height:14px;border:2px solid #FFD700"></span> Box Cứu Hộ</div>
                </div>
                """)
                gr.Markdown("<br>")
                btn_finish = gr.Button("🏁 KẾT THÚC KHÁM BỆNH & ĐẶT LẠI", variant="stop")

    # =========================================================
    # LIÊN KẾT SỰ KIỆN (EVENT WIRING)
    # =========================================================

    # Bước 1
    btn_predict_cls.click(run_classification, inputs=[input_image], outputs=[class_output])
    btn_nosick.click(confirm_nosick, outputs=[html_alert, btn_go_seg])
    btn_sick.click(confirm_sick, outputs=[html_alert, btn_go_seg])

    # Nút chuyển bước 1 -> 2
    btn_go_seg.click(
        switch_to_segmentation,
        inputs=[input_image],
        outputs=[group_ctrl_cls, group_out_cls, group_ctrl_seg, group_out_seg, saved_original_image, points_state, current_step, status_box_seg]
    )

    # Bước 2
    input_image.select(get_clicks, [input_image, points_state, current_step], [input_image, points_state, status_box_seg])
    btn_predict_seg.click(run_segmentation, [input_image, points_state, saved_original_image], [output_image_seg, status_box_seg])
    btn_reset_seg.click(reset_prompt, [saved_original_image], [input_image, points_state, status_box_seg])

    # Nút Kết thúc
    btn_finish.click(
        switch_to_classification,
        outputs=[group_ctrl_cls, group_out_cls, group_ctrl_seg, group_out_seg, input_image, output_image_seg, current_step, class_output, html_alert, btn_go_seg]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=False)