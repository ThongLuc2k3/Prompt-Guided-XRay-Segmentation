import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptCrossAttention2D(nn.Module):
    def __init__(self, img_channels, prompt_channels, embed_dim):
        super(PromptCrossAttention2D, self).__init__()
        
        # 1. Bộ tạo Query (Từ Prompt Box)
        self.W_q = nn.Conv2d(prompt_channels, embed_dim, kernel_size=1)
        
        # 2. Bộ tạo Key và Value (Từ Ảnh X-quang)
        self.W_k = nn.Conv2d(img_channels, embed_dim, kernel_size=1)
        self.W_v = nn.Conv2d(img_channels, embed_dim, kernel_size=1)
        
        # Hệ số scale (chuẩn hóa giống bài báo Transformer)
        self.scale = embed_dim ** -0.5
        
        # Lớp Conv cuối cùng để mượt mà hóa đầu ra
        self.out_conv = nn.Conv2d(embed_dim, img_channels, kernel_size=1)

    def forward(self, img_features, prompt_features):
        B, C, H, W = img_features.shape
        N = H * W # Tổng số pixel
        
        # Bước 1: Chiếu (Projection) và Duỗi phẳng (Flatten) không gian
        # Q: [B, N, Embed_Dim]
        Q = self.W_q(prompt_features).view(B, -1, N).permute(0, 2, 1) 
        
        # K: [B, Embed_Dim, N]
        K = self.W_k(img_features).view(B, -1, N)
        
        # V: [B, N, Embed_Dim]
        V = self.W_v(img_features).view(B, -1, N).permute(0, 2, 1)

        # Bước 2: Dot-Product (Thương lượng niềm tin)
        # Q nhân với K^T -> Ma trận Attention [B, N, N]
        attn_scores = torch.bmm(Q, K) * self.scale
        
        # Hàm Softmax quyết định "Tin bao nhiêu phần" (0.0 đến 1.0)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Bước 3: Áp dụng niềm tin vào Thông tin gốc (Value)
        # [B, N, N] x [B, N, Embed_Dim] -> [B, N, Embed_Dim]
        attended_features = torch.bmm(attn_probs, V)
        
        # Bước 4: Khôi phục lại hình dạng ảnh 2D [B, Embed_Dim, H, W]
        attended_features = attended_features.permute(0, 2, 1).contiguous().view(B, -1, H, W)
        
        # Cộng phần dư (Residual Connection) để tránh mất thông tin gốc
        out = self.out_conv(attended_features) + img_features
        
        return out