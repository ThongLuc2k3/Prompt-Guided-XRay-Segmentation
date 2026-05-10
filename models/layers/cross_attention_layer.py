import torch
import torch.nn as nn

class PromptCrossAttention2D(nn.Module):
    def __init__(self, img_channels, prompt_channels, embed_dim, window_size=16):
        super(PromptCrossAttention2D, self).__init__()
        
        # Kích thước "lát cắt" (Window Size) giống hệt Swin Transformer
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        # Bộ tạo Q, K, V
        self.W_q = nn.Conv2d(prompt_channels, embed_dim, kernel_size=1)
        self.W_k = nn.Conv2d(img_channels, embed_dim, kernel_size=1)
        self.W_v = nn.Conv2d(img_channels, embed_dim, kernel_size=1)
        
        self.scale = embed_dim ** -0.5
        self.out_conv = nn.Conv2d(embed_dim, img_channels, kernel_size=1)

    def window_partition(self, x):
        """Băm bức ảnh thành các cửa sổ nhỏ độc lập"""
        B, C, H, W = x.shape
        # Chia H và W thành các block kích thước window_size
        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        # Gộp các block lại: [Số_lượng_cửa_sổ_tổng, Số_pixel_1_cửa_sổ, C]
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size**2, C)
        return windows

    def window_reverse(self, windows, H, W, B):
        """Lắp ráp các cửa sổ lại thành bức ảnh ban đầu"""
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, self.embed_dim)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, self.embed_dim, H, W)
        return x

    def forward(self, img_features, prompt_features):
        B, C, H, W = img_features.shape
        
        # Trích xuất đặc trưng
        Q_full = self.W_q(prompt_features)
        K_full = self.W_k(img_features)
        V_full = self.W_v(img_features)

        # =================================================================
        # 🔥 ĐỘT PHÁ TOÁN HỌC: SWIN WINDOW ATTENTION 🔥
        # Băm nhỏ tất cả thành các lát cắt 16x16 để giải phóng VRAM
        # =================================================================
        # Shape sau khi băm: [B * Số_cửa_sổ, 256, Embed_Dim]
        q_win = self.window_partition(Q_full)
        k_win = self.window_partition(K_full)
        v_win = self.window_partition(V_full)

        # Tính Attention nội bộ trong từng lát cắt nhỏ xíu (Tính toán song song siêu nhanh)
        # [Num_Windows, 256, Embed_Dim] x [Num_Windows, Embed_Dim, 256] -> [Num_Windows, 256, 256]
        attn_scores = torch.bmm(q_win, k_win.transpose(1, 2)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Áp dụng niềm tin vào Value
        attended_win = torch.bmm(attn_probs, v_win)

        # Lắp ráp các mảnh vỡ lại thành bức ảnh lớn
        attended_features = self.window_reverse(attended_win, H, W, B)
        
        # Cộng Residual Connection (Skip connection để giữ nét)
        out = self.out_conv(attended_features) + img_features
        
        return out