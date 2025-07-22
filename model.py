import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        x = x.permute(2, 3, 0, 1).flatten(0, 1)  # [H*W, B, C]
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        h = w = int(x.shape[0] ** 0.5)
        x = x.view(h, w, -1, x.size(-1)).permute(2, 3, 0, 1)
        return x


class UNetViT(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, vit_embed_dim=512, vit_heads=8):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, vit_embed_dim)

        self.pool = nn.MaxPool2d(2)

        self.vit = ViTBlock(vit_embed_dim, vit_heads)

        self.up4 = nn.ConvTranspose2d(vit_embed_dim, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        v = self.vit(e4)

        d4 = self.up4(v)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return torch.sigmoid(self.final_conv(d2))