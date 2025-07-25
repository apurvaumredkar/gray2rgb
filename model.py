import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        quarter = out_channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, quarter, 1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, quarter, 1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
            nn.Conv2d(quarter, quarter, 3, padding=1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, quarter, 1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
            nn.Conv2d(quarter, quarter, 3, padding=1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
            nn.Conv2d(quarter, quarter, 3, padding=1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, quarter, 1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        ], dim=1)

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0,2,1)  
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        x = x.permute(0,2,1).view(B, C, H, W)
        return x

class UNetViT(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, vit_embed_dim=256, vit_heads=4):
        super().__init__()
        
        self.enc1 = InceptionBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)          
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)          
        self.enc3 = InceptionBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)          
        self.enc4 = ConvBlock(128, vit_embed_dim)
        
        self.vit = ViTBlock(vit_embed_dim, vit_heads)  

        self.up4 = nn.ConvTranspose2d(vit_embed_dim, 128, 2, stride=2)     
        self.dec4 = ConvBlock(128 + 128, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)                
        self.dec3 = ConvBlock(64 + 64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)                 
        self.dec2 = ConvBlock(32 + 32, 32)

        self.final_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        
        e1 = self.enc1(x)               
        p1 = self.pool1(e1)             
        e2 = self.enc2(p1)              
        p2 = self.pool2(e2)             
        e3 = self.enc3(p2)              
        p3 = self.pool3(e3)             
        e4 = self.enc4(p3)              
        
        v = self.vit(e4)                

        d4 = self.up4(v)                
        d4 = self.dec4(torch.cat([d4, e3], dim=1))  
        d3 = self.up3(d4)               
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  
        d2 = self.up2(d3)               
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  
        out = self.final_conv(d2)   
        return out

if __name__ == "__main__":
    from torchinfo import summary
    import json
    with open("hyperparameters.json", "r") as f:
        hparams = json.load(f)
    vit_embed_dim = hparams.get("vit_embed_dim", 512)
    vit_heads = hparams.get("vit_heads", 8)
    model = UNetViT(in_channels=1, out_channels=2,
                    vit_embed_dim=vit_embed_dim, vit_heads=vit_heads)
    print(summary(model, input_size=(1,1,256,256)))
