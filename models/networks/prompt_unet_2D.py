import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks_other import init_weights
from .utils import unetConv2
from models.layers.grid_attention_layer import GridAttentionBlock2D

# ================================================================
# 1. KHỐI UPSAMPLING TÍCH HỢP PROMPT VÀ ATTENTION
# ================================================================
class unetUp_PromptAttention(nn.Module):
    def __init__(self, skip_channels, gating_channels, out_channels):
        super(unetUp_PromptAttention, self).__init__()
        
        # 1. Cổng Attention chuẩn (ĐÃ SỬA LỖI 3D CỦA TÁC GIẢ)
        self.attention = GridAttentionBlock2D(in_channels=skip_channels, 
                                              gating_channels=gating_channels, 
                                              inter_channels=skip_channels // 2,
                                              sub_sample_factor=(2, 2)) # <--- THÊM DÒNG NÀY
        
        # 2. Bộ mã hóa Prompt (Prompt Encoder)
        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(1, gating_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(gating_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. Mạng giải mã chuẩn
        self.up = nn.ConvTranspose2d(gating_channels, skip_channels, kernel_size=4, stride=2, padding=1)
        self.conv = unetConv2(skip_channels * 2, out_channels, is_batchnorm=True)
        
    def forward(self, skip, gating, prompt):
        # 1. Resize Prompt Heatmap về bằng kích thước của tín hiệu Gating
        p_resized = F.interpolate(prompt, size=gating.shape[2:], mode='bilinear', align_corners=False)
        
        # 2. Mã hóa Prompt
        p_encoded = self.prompt_encoder(p_resized)
        
        # 3. FUSION: Trộn Prompt vào Tín hiệu Gating
        g_fused = gating + p_encoded
        
        # 4. Lọc Skip Connection qua cổng Attention (bị điều khiển bởi G_fused)
        skip_att = self.attention(skip, g_fused)
        
        # Xử lý trường hợp code gốc trả về tuple (gated_feature, attention_map)
        if isinstance(skip_att, tuple):
            skip_att = skip_att[0]
            
        # 5. Upsample tín hiệu Gating GỐC (giữ nguyên luồng ngữ nghĩa)
        up_gating = self.up(gating)
        
        # Căn chỉnh kích thước nếu bị lệch pixel do padding
        offset = up_gating.size()[2] - skip_att.size()[2]
        pad = 2 * [offset // 2, offset // 2]
        skip_att = F.pad(skip_att, pad)
        
        # 6. Nối (Concat) và Conv như U-Net chuẩn
        out = self.conv(torch.cat([skip_att, up_gating], dim=1))
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