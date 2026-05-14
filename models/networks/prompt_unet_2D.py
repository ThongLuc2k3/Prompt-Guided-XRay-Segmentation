import torch
import torch.nn as nn
import torch.nn.functional as F

# Khối Conv cơ bản cho U-Net
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU(inplace=True))

    def forward(self, inputs):
        return self.conv2(self.conv1(inputs))

class unetUp_PromptAttention(nn.Module):
    def __init__(self, skip_channels, gating_channels, out_channels, prompt_weight=1.0):
        super(unetUp_PromptAttention, self).__init__()
        
        self.alpha_raw = nn.Parameter(torch.tensor(-0.84)) 
        self.w = prompt_weight
        self.beta = nn.Parameter(torch.tensor(0.05)) # Learnable Boundary Refinement
        
        from models.layers.grid_attention_layer import GridAttentionBlock2D
        self.attention = GridAttentionBlock2D(
            in_channels=skip_channels, 
            gating_channels=gating_channels, 
            inter_channels=skip_channels // 2,
            sub_sample_factor=(2, 2)
        )
        
        # Thêm InstanceNorm2d để giữ Intensity Prior ổn định
        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(1, gating_channels // 2, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(gating_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(gating_channels // 2, gating_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(gating_channels),
            nn.ReLU(inplace=True)
        )
        
        self.prompt_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gating_channels, 1, 1),
            nn.Sigmoid()
        )
        
        self.up = nn.ConvTranspose2d(gating_channels, skip_channels, kernel_size=4, stride=2, padding=1)
        self.conv = unetConv2(skip_channels * 2, out_channels, is_batchnorm=True)
        
    def forward(self, skip, gating, prompt):
        p_encoded = self.prompt_encoder(prompt if prompt.shape[2:] == gating.shape[2:] else 
                                       F.interpolate(prompt, size=gating.shape[2:], mode='bilinear'))
        
        conf = self.prompt_confidence(p_encoded) 
        alpha = torch.sigmoid(self.alpha_raw)
        
        g_fused = gating + (conf * alpha * self.w * p_encoded)
        
        skip_att = self.attention(skip, g_fused)
        if isinstance(skip_att, tuple): skip_att = skip_att[0]
        skip_att = skip_att + 0.3 * skip
        
        up_gating = self.up(gating)
        
        diffY, diffX = up_gating.size()[2] - skip_att.size()[2], up_gating.size()[3] - skip_att.size()[3]
        skip_att = F.pad(skip_att, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        out = self.conv(torch.cat([skip_att, up_gating], dim=1))
        
        # Boundary Refinement tự động điều chỉnh
        p_refine = F.interpolate(prompt, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = out + (self.beta * p_refine)
        
        return out

class PGA_UNet(nn.Module):
    def __init__(self, feature_scale=4, n_classes=1, in_channels=1, is_batchnorm=True):
        super(PGA_UNet, self).__init__()
        
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]

        # Downsampling
        self.conv1 = unetConv2(in_channels, filters[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], is_batchnorm)

        # Upsampling với Prompt Guided Attention
        self.up_concat4 = unetUp_PromptAttention(filters[3], filters[4], filters[3], prompt_weight=1.0)
        self.up_concat3 = unetUp_PromptAttention(filters[2], filters[3], filters[2], prompt_weight=0.7)
        self.up_concat2 = unetUp_PromptAttention(filters[1], filters[2], filters[1], prompt_weight=0.4)
        self.up_concat1 = unetUp_PromptAttention(filters[0], filters[1], filters[0], prompt_weight=0.2)

        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs, prompt):
        # Augmentation cấp độ mô hình (Độc lập & An toàn)
        if self.training:
            r = torch.rand(1).item()
            if r < 0.15: # Mất prompt hoàn toàn
                prompt = torch.zeros_like(prompt)
            elif r < 0.30: # Bác sĩ run tay (Nhiễu)
                prompt = prompt + torch.randn_like(prompt) * 0.1
                prompt = torch.clamp(prompt, 0, 1)

        c1 = self.conv1(inputs)
        c2 = self.conv2(self.maxpool1(c1))
        c3 = self.conv3(self.maxpool2(c2))
        c4 = self.conv4(self.maxpool3(c3))
        center = self.center(self.maxpool4(c4))

        up4 = self.up_concat4(c4, center, prompt)
        up3 = self.up_concat3(c3, up4, prompt)
        up2 = self.up_concat2(c2, up3, prompt)
        up1 = self.up_concat1(c1, up2, prompt)

        return self.final(up1)