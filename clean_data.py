import os
import cv2
import glob

def clean_png_icc_profile(base_dir):
    print(f"🧹 Đang quét tìm và rửa ảnh PNG trong thư mục: {base_dir}")
    
    # Tìm TẤT CẢ các file .png trong thư mục dataset (bao gồm cả thư mục con train/val/test)
    search_pattern = os.path.join(base_dir, "**", "*.png")
    filepaths = glob.glob(search_pattern, recursive=True)
    
    count = 0
    for path in filepaths:
        try:
            # 1. Đọc ảnh vào bộ nhớ (giữ nguyên gốc)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            
            if img is not None:
                # 2. Ghi đè lại ảnh đúng vị trí cũ
                # Lúc này OpenCV sẽ lưu theo chuẩn PNG sạch, không chứa ICC Profile tạp nham
                cv2.imwrite(path, img)
                count += 1
        except Exception as e:
            print(f"Lỗi ở file {path}: {e}")

    print("="*50)
    print(f"✅ HOÀN TẤT! Đã rửa sạch {count} tấm ảnh.")
    print("="*50)

if __name__ == "__main__":
    # Chỉ định đúng thư mục chứa data của bạn
    DATASET_FOLDER = "dataset_BTXRD" 
    
    clean_png_icc_profile(DATASET_FOLDER)