import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x)

        x = self.dropout(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  
        
        # Self-attention with residual
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x = self.norm1(x_flat + self.dropout(attn_out))
        
        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

class AttentionGate(nn.Module):
    """Attention gate for skip connections to reduce redundant features"""
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.skip_conv = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.attention_conv = nn.Conv2d(inter_channels, 1, kernel_size=1)
        
    def forward(self, gate, skip):
        g = self.gate_conv(gate)
        s = self.skip_conv(skip)
        attention = torch.sigmoid(self.attention_conv(F.relu(g + s)))
        return skip * attention

class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization for style transfer capabilities"""
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        
    def forward(self, x):
        return self.norm(x)

class ColorizationNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, dropout=0.1, 
                 vit_embed_dim=128, vit_heads=8, num_vit_layers=3):
        super().__init__()

        # Encoder with residual connections
        self.enc1 = ConvBlock(in_channels, 32, dropout)    
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64, dropout)             
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128, dropout)             
        self.pool3 = nn.MaxPool2d(2)

        # Bridge to ViT
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, vit_embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, vit_embed_dim), vit_embed_dim),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Multiple ViT blocks for better global understanding
        self.vit_layers = nn.ModuleList([
            ViTBlock(vit_embed_dim, vit_heads, dropout) 
            for _ in range(num_vit_layers)
        ])

        # Attention gates for skip connections
        self.att_gate3 = AttentionGate(128, 128, 64)  # Fixed: both inputs are 128 channels
        self.att_gate2 = AttentionGate(64, 64, 32)    # Fixed: both inputs are 64 channels  
        self.att_gate1 = AttentionGate(32, 32, 16)    # Fixed: both inputs are 32 channels

        # Decoder with attention gates
        self.up3 = nn.ConvTranspose2d(vit_embed_dim, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128, dropout)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64, dropout)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32 + 32, 32, dropout)

        # Enhanced final layers with multiple outputs
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, 8),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        # Separate heads for different aspects
        self.color_head = nn.Conv2d(8, out_channels, kernel_size=1)
        self.confidence_head = nn.Conv2d(8, 1, kernel_size=1)
        
        # Adaptive normalization for final output
        self.adaptive_norm = AdaptiveInstanceNorm(out_channels)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)

        # Multiple ViT layers for better global context
        vit_out = e4
        for vit_layer in self.vit_layers:
            vit_out = vit_layer(vit_out)

        # Decoder with attention gates
        d3 = self.up3(vit_out)
        e3_att = self.att_gate3(d3, e3)  # Both d3 and e3 are 128 channels
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att_gate2(d2, e2)  # Both d2 and e2 are 64 channels
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att_gate1(d1, e1)  # Both d1 and e1 are 32 channels
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        # Final processing
        features = self.final_conv(d1)
        
        # Color prediction
        colors = self.color_head(features)
        colors = torch.tanh(colors) * 1.2  # Slightly expand range
        
        # Confidence prediction for uncertainty-aware training
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return colors, confidence



if __name__ == "__main__":
    from torchinfo import summary
    import json

    # Default hyperparameters for RTX 4070 laptop
    default_hparams = {
        "resolution": 256,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "dropout": 0.15,
        "vit_embed_dim": 256,
        "vit_heads": 8,
        "num_vit_layers": 2,
        "weight_decay": 1e-5,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "total_epochs": 100
    }

    try:
        with open("hyperparameters.json", "r") as f:
            hparams = json.load(f)
    except FileNotFoundError:
        hparams = default_hparams
        print("Using default hyperparameters")

    # Update with defaults if keys missing
    for key, value in default_hparams.items():
        hparams.setdefault(key, value)

    model = ColorizationNet(
        in_channels=1,
        out_channels=2,
        dropout=hparams["dropout"],
        vit_embed_dim=hparams["vit_embed_dim"],
        vit_heads=hparams["vit_heads"],
        num_vit_layers=hparams["num_vit_layers"]
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(summary(model, input_size=(hparams["batch_size"], 1, hparams["resolution"], hparams["resolution"])))