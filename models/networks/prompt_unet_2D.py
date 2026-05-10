import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks_other import init_weights
from .utils import unetConv2
from models.layers.grid_attention_layer import GridAttentionBlock2D

# ================================================================
# 1. KHỐI UPSAMPLING TÍCH HỢP PROMPT VÀ Swin Transformer
# ================================================================
from models.layers.cross_attention_layer import PromptCrossAttention2D # Nhớ import file vừa tạo

class unetUp_PromptAttention(nn.Module):
    def __init__(self, skip_channels, gating_channels, out_channels):
        super(unetUp_PromptAttention, self).__init__()
        
        # THAY THẾ Ở ĐÂY: Dùng Cross-Attention thay vì Attention Gate tuyến tính
        # Gating/Skip đóng vai trò Image (img_channels), Prompt là prompt_channels
        self.cross_attention = PromptCrossAttention2D(
            img_channels=skip_channels, 
            prompt_channels=1, # Prompt đầu vào chỉ có 1 kênh (Heatmap)
            embed_dim=skip_channels // 2
        )
        
        # Bộ Upsampling chuẩn
        self.up = nn.ConvTranspose2d(gating_channels, skip_channels, kernel_size=4, stride=2, padding=1)
        self.conv = unetConv2(skip_channels * 2, out_channels, is_batchnorm=True)

    def forward(self, skip, gating, prompt):
        # 1. Thu nhỏ Prompt cho khớp kích thước Skip
        p_resized = F.interpolate(prompt, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # 2. 🔥 GIAO TRANH Q-K-V 🔥
        # Để Skip (đặc trưng ảnh sắc nét) làm Image, Prompt làm Query
        skip_attended = self.cross_attention(img_features=skip, prompt_features=p_resized)
        
        # 3. Phóng to Gating từ dưới lên
        up_gating = self.up(gating)
        
        # 4. Nối và Conv
        out = self.conv(torch.cat([skip_attended, up_gating], dim=1))
        return out

# ================================================================
# 2. TOÀN BỘ KIẾN TRÚC MẠNG PROMPT-GUIDED ATTENTION U-NET
# ================================================================
class Prompt_Att_UNet_2D(nn.Module):
    def __init__(self, feature_scale=4, n_classes=1, in_channels=1, is_batchnorm=True):
        super(Prompt_Att_UNet_2D, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters] # [16, 32, 64, 128, 256]

        # --- DOWNSAMPLING (Giữ nguyên như gốc) ---
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # --- UPSAMPLING (Đã thay bằng khối PromptAttention) ---
        self.up_concat4 = unetUp_PromptAttention(skip_channels=filters[3], gating_channels=filters[4], out_channels=filters[3])
        self.up_concat3 = unetUp_PromptAttention(skip_channels=filters[2], gating_channels=filters[3], out_channels=filters[2])
        self.up_concat2 = unetUp_PromptAttention(skip_channels=filters[1], gating_channels=filters[2], out_channels=filters[1])
        self.up_concat1 = unetUp_PromptAttention(skip_channels=filters[0], gating_channels=filters[1], out_channels=filters[0])

        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # Khởi tạo trọng số
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs, prompt):
        # Đi xuống
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        # Đi lên: Truyền thẳng Prompt vào 4 cái Cửa Chú Ý!
        up4 = self.up_concat4(skip=conv4, gating=center, prompt=prompt)
        up3 = self.up_concat3(skip=conv3, gating=up4, prompt=prompt)
        up2 = self.up_concat2(skip=conv2, gating=up3, prompt=prompt)
        up1 = self.up_concat1(skip=conv1, gating=up2, prompt=prompt)

        final = self.final(up1)
        return final