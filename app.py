import gradio as gr
import torch
import numpy as np
import cv2
import warnings

# Bỏ qua các cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# Import Models
from models.networks.unet_2D import unet_2D
from models.networks.prompt_unet_2D import Prompt_Att_UNet_2D

# =========================================================
# 1. CẤU HÌNH & TẢI TRỌNG SỐ (LOAD MODELS)
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512

print(f"[*] Đang khởi tạo hệ thống trên thiết bị: {DEVICE}")

# 1.1 Khởi tạo Baseline Model
model_unet = unet_2D(in_channels=1, n_classes=1).to(DEVICE)
try:
    model_unet.load_state_dict(torch.load("checkpoints/Unet_2D/Unet2D_best.pth", map_location=DEVICE))
    model_unet.eval()
    print("[+] Tải thành công trọng số: Unet2D_best.pth")
except Exception as e:
    print(f"[!] Cảnh báo: Không tìm thấy trọng số Unet2D_best.pth. Lỗi: {e}")

# 1.2 Khởi tạo Prompt Model
model_prompt = Prompt_Att_UNet_2D(in_channels=1, n_classes=1).to(DEVICE)
try:
    model_prompt.load_state_dict(torch.load("checkpoints/Prompt_ATT_Unet2D/Prompt_ATT_Unet2D_best.pth", map_location=DEVICE))
    model_prompt.eval()
    print("[+] Tải thành công trọng số: Prompt_ATT_Unet2D_best.pth")
except Exception as e:
    print(f"[!] Cảnh báo: Không tìm thấy trọng số Prompt_ATT_Unet2D_best.pth. Lỗi: {e}")


# =========================================================
# 2. HÀM PHỤ TRỢ (HELPERS)
# =========================================================
def create_plateau_heatmap(bbox, orig_h, orig_w):
    """Hàm tạo Plateau Heatmap chuẩn như lúc Train"""
    heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
    x_min, y_min, x_max, y_max = bbox
    
    padding = 5
    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
    x_max, y_max = min(orig_w, x_max + padding), min(orig_h, y_max + padding)

    heatmap[int(y_min):int(y_max), int(x_min):int(x_max)] = 1.0
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    return heatmap

def overlay_prediction(original_img, mask, prompt_bbox=None):
    """Phủ màu lên ảnh: Khối u màu Đỏ (Red), Bounding Box màu Vàng (Yellow)"""
    # Nếu ảnh đầu vào là đen trắng (2D), chuyển sang RGB (3D) để vẽ màu
    if len(original_img.shape) == 2:
        overlay = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        overlay = original_img.copy()

    # Phủ màu đỏ cho mask
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask == 1] = [255, 0, 0] # Đỏ [R, G, B]
    
    # Kết hợp ảnh gốc và mask (alpha = 0.5)
    output = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

    # Nếu có box prompt, vẽ viền vàng
    if prompt_bbox is not None:
        x_min, y_min, x_max, y_max = prompt_bbox
        cv2.rectangle(output, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 0), 2)
        
    return output


# =========================================================
# 3. LOGIC XỬ LÝ SỰ KIỆN GIAO DIỆN
# =========================================================
def get_clicks(img, evt: gr.SelectData, points_state):
    """Bắt sự kiện click chuột trên ảnh để lấy 2 điểm"""
    if img is None:
        return img, points_state, "Vui lòng tải ảnh lên trước!"
        
    # Thêm tọa độ (x, y) vừa click vào bộ nhớ trạng thái
    points_state.append((evt.index[0], evt.index[1]))
    
    # Vẽ điểm đỏ lên ảnh để báo hiệu cho người dùng
    img_drawn = img.copy()
    for p in points_state:
        cv2.circle(img_drawn, p, radius=5, color=(255, 0, 0), thickness=-1)
        
    # Nếu đã đủ 2 điểm, vẽ luôn cái Box viền xanh lá
    if len(points_state) >= 2:
        x1, y1 = points_state[-2]
        x2, y2 = points_state[-1]
        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img_drawn, points_state, f"Đã nhận 2 điểm. Bạn có thể bấm Dự Đoán!"
        
    return img_drawn, points_state, f"Đã lấy 1 điểm: {points_state[-1]}. Hãy chấm điểm thứ 2!"


def run_inference(image, model_choice, points_state):
    """Hàm chạy AI sinh ra kết quả"""
    if image is None:
        return None, "❌ Lỗi: Chưa có ảnh đầu vào."
        
    orig_h, orig_w = image.shape[:2]
    
    # Ép về ảnh xám để model đọc
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = image

    # Preprocess
    img_resized = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(DEVICE)

    prompt_bbox = None

    with torch.no_grad():
        if model_choice == "U-Net 2D (Baseline)":
            outputs = model_unet(img_tensor)
            
        else: # Prompt_Att_UNet_2D
            if len(points_state) < 2:
                return None, "⚠️ Lỗi: Bạn chọn Prompt Model nhưng chưa chấm đủ 2 điểm trên ảnh!"
            
            # Lấy 2 điểm cuối cùng trong state
            x1, y1 = points_state[-2]
            x2, y2 = points_state[-1]
            prompt_bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            
            # Tạo heatmap
            heatmap = create_plateau_heatmap(prompt_bbox, orig_h, orig_w)
            heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
            heatmap_tensor = torch.from_numpy(heatmap_resized).unsqueeze(0).unsqueeze(0).float()
            heatmap_tensor = heatmap_tensor.to(DEVICE)
            
            # Predict
            outputs = model_prompt(img_tensor, heatmap_tensor)

    # Post-process (Lấy Threshold 0.5)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float().squeeze().cpu().numpy()
    
    # Phóng to mask về lại kích thước ảnh gốc
    pred_orig = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Phủ màu lên ảnh gốc
    result_img = overlay_prediction(image, pred_orig, prompt_bbox)

    return result_img, "✅ Hoàn tất! Kết quả phân đoạn đã được hiển thị."

def reset_all():
    """Reset trạng thái về số 0"""
    return None, None, [], "Đã reset! Vui lòng tải ảnh mới lên."

# =========================================================
# 4. GIAO DIỆN CHÍNH CỦA GRADIO (UI LAYOUT)
# =========================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 DEMO: PROMPT-GUIDED ATTENTION U-NET")
    gr.Markdown("Hệ thống Phân đoạn X-quang Xương tương tác thông minh. Chọn model và thao tác bên dưới.")
    
    # Lưu tọa độ click chuột
    points_state = gr.State([])

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Ảnh Gốc (Upload & Click)")
            model_radio = gr.Radio(["U-Net 2D (Baseline)", "Prompt Attention U-Net 2D"], 
                                   value="Prompt Attention U-Net 2D", 
                                   label="Mô hình Dự đoán")
            
            status_text = gr.Textbox(label="Trạng thái hệ thống", value="Đang chờ dữ liệu...", interactive=False)
            
            with gr.Row():
                btn_run = gr.Button("🧠 Dự Đoán", variant="primary")
                btn_clear = gr.Button("🗑️ Reset", variant="secondary")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Kết Quả Phân Đoạn")

    # BẮT SỰ KIỆN
    # Khi user click vào ảnh gốc -> Gọi hàm get_clicks
    input_image.select(
        fn=get_clicks, 
        inputs=[input_image, points_state], 
        outputs=[input_image, points_state, status_text]
    )

    # Khi nhấn nút Dự đoán
    btn_run.click(
        fn=run_inference,
        inputs=[input_image, model_radio, points_state],
        outputs=[output_image, status_text]
    )

    # Khi nhấn nút Reset
    btn_clear.click(
        fn=reset_all,
        inputs=[],
        outputs=[input_image, output_image, points_state, status_text]
    )

if __name__ == "__main__":
    # Khởi chạy server
    demo.launch(share=True, debug=True)