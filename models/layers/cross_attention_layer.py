import torch
import torch.nn as nn

class PromptCrossAttention2D(nn.Module):
    def __init__(self, img_channels, prompt_channels, embed_dim, window_size=16):
        super(PromptCrossAttention2D, self).__init__()
        
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        # =========================================================
        # 1. BỘ CHUYỂN ĐỔI (PROJECTION) MỚI
        # =========================================================
        # Ảnh là chủ thể chính (Tạo ra cả Q, K, V)
        self.W_q_img = nn.Conv2d(img_channels, embed_dim, kernel_size=1)
        self.W_k = nn.Conv2d(img_channels, embed_dim, kernel_size=1)
        self.W_v = nn.Conv2d(img_channels, embed_dim, kernel_size=1)
        
        # Prompt chỉ đóng vai trò là "Chất xúc tác" (Modulator) cho Query
        self.W_p = nn.Conv2d(prompt_channels, embed_dim, kernel_size=1)
        
        self.scale = embed_dim ** -0.5
        self.out_conv = nn.Conv2d(embed_dim, img_channels, kernel_size=1)

    def window_partition(self, x):
        """Băm bức ảnh thành các lát cắt để tiết kiệm VRAM"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size**2, C)
        return windows

    def window_reverse(self, windows, H, W, B):
        """Lắp ráp các lát cắt lại thành ảnh gốc"""
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, self.embed_dim)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, self.embed_dim, H, W)
        return x

    def forward(self, img_features, prompt_features):
        B, C, H, W = img_features.shape
        
        # =========================================================
        # 2. 🔥 TRIẾT LÝ: PROMPT CHỈ LÀ GỢI Ý (HINT BIAS) 🔥
        # =========================================================
        # Query được hình thành từ Đặc trưng Ảnh CỘNG VỚI Gợi ý của Prompt
        Q_img = self.W_q_img(img_features)
        Q_prompt = self.W_p(prompt_features)
        Q_full = Q_img + Q_prompt  # Dung hợp! Ảnh vẫn là gốc.
        
        # Key và Value hoàn toàn là của Ảnh (Ảnh giữ quyền quyết định cuối cùng)
        K_full = self.W_k(img_features)
        V_full = self.W_v(img_features)

        # =========================================================
        # 3. CƠ CHẾ SWIN WINDOW ATTENTION (Tránh OOM)
        # =========================================================
        q_win = self.window_partition(Q_full)
        k_win = self.window_partition(K_full)
        v_win = self.window_partition(V_full)

        # Tính toán niềm tin nội bộ trong từng ô 16x16
        attn_scores = torch.bmm(q_win, k_win.transpose(1, 2)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Áp dụng niềm tin
        attended_win = torch.bmm(attn_probs, v_win)

        # Khôi phục ảnh
        attended_features = self.window_reverse(attended_win, H, W, B)
        
        # Skip connection để không bao giờ quên đặc trưng gốc
        out = self.out_conv(attended_features) + img_features
        
        return out