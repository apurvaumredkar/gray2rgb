import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import json
from torch.nn.utils import spectral_norm
from torchinfo import summary


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                            nn.GroupNorm(8, out_channels),
                            nn.LeakyReLU(0.01, inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.GroupNorm(8, out_channels),
                            nn.LeakyReLU(0.01, inplace=True),
                        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv_block(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.ag = AttentionGate(F_g=in_channels // 2, F_l=skip_channels, F_int=in_channels // 4)

        conv_in_channels = in_channels // 2 + skip_channels
        
        self.conv_block = nn.Sequential(
                            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1),
                            nn.GroupNorm(8, out_channels),
                            nn.LeakyReLU(0.01, inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.GroupNorm(8, out_channels),
                            nn.LeakyReLU(0.01, inplace=True),
                        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.ag(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x
    
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
                    nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GroupNorm(8, F_int),
                )
        self.W_x = nn.Sequential(
                    nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GroupNorm(8, F_int),
                )
        self.psi = nn.Sequential(
                    nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GroupNorm(1, 1),
                    nn.Sigmoid(),
                )
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ViTUNetColorizer(nn.Module):
    def __init__(self, vit_model_name="vit_tiny_patch16_224", freeze_vit_epochs=10):
        super(ViTUNetColorizer, self).__init__()

        self.vit = timm.create_model(vit_model_name, pretrained=True, num_classes=0)
        self.vit_embed_dim = self.vit.embed_dim
        self.vit.head = nn.Identity()

        self.enc1 = EncoderBlock(1, 16)
        self.enc2 = EncoderBlock(16, 32)
        self.enc3 = EncoderBlock(32, 64)
        self.enc4 = EncoderBlock(64, 128)

        self.bottleneck_processor = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.AdaptiveAvgPool2d((14, 14)),
        )

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(128 + self.vit_embed_dim, 128, kernel_size=1), # type: ignore
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.dec4 = DecoderBlock(128, 64, 64)
        self.dec3 = DecoderBlock(64, 32, 32)
        self.dec2 = DecoderBlock(32, 16, 16)

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, 8),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(8, 2, kernel_size=1),
            nn.Tanh(),
        )

        self.freeze_vit_epochs = freeze_vit_epochs
        self.current_epoch = 0

    def extract_vit_features(self, x):
        B = x.shape[0]
        x_3ch = x.repeat(1, 3, 1, 1)

        if x_3ch.shape[-1] != 224:
            x_3ch = F.interpolate(
                x_3ch, size=(224, 224), mode="bicubic", align_corners=False
            )

        x_vit = self.vit.patch_embed(x_3ch) # type: ignore
        if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
            x_vit = x_vit + self.vit.pos_embed[:, 1:, :] # type: ignore
        x_vit = self.vit.pos_drop(x_vit) # type: ignore

        for block in self.vit.blocks: # type: ignore
            x_vit = block(x_vit)

        x_vit = self.vit.norm(x_vit) # type: ignore
        x_vit = x_vit.transpose(1, 2).reshape(B, self.vit_embed_dim, 14, 14)

        return x_vit

    def forward(self, x):

        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)

        bottleneck = self.bottleneck_processor(x4)
        vit_features = self.extract_vit_features(x)
        fused = torch.cat([bottleneck, vit_features], dim=1)
        fused = self.fusion_layer(fused)

        fused = F.interpolate(fused, size=x3.shape[2:], mode="bilinear", align_corners=False)

        d4 = self.dec4(fused, skip3)
        d3 = self.dec3(d4, skip2)
        d2 = self.dec2(d3, skip1)

        out = self.final_conv(d2)

        return out

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        requires_grad = epoch >= self.freeze_vit_epochs
        for param in self.vit.parameters():
            param.requires_grad = requires_grad

    def get_param_groups(self, lr_decoder=1e-4, lr_vit=1e-5):
        vit_params = []
        decoder_params = []
        for name, param in self.named_parameters():
            if "vit" in name:
                vit_params.append(param)
            else:
                decoder_params.append(param)
        return [
            {"params": decoder_params, "lr": lr_decoder},
            {"params": vit_params, "lr": lr_vit},
        ]


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, n_filters=64):
        super(PatchDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2):
            return [
                spectral_norm(
                    nn.Conv2d(
                        in_filters, out_filters, kernel_size=4, stride=stride, padding=1
                    )
                ),
                nn.LeakyReLU(0.01, inplace=True)
            ]

        self.model = nn.Sequential(
            *discriminator_block(in_channels, n_filters),
            *discriminator_block(n_filters, n_filters * 2),
            *discriminator_block(n_filters * 2, n_filters * 4),
            spectral_norm(nn.Conv2d(n_filters * 4, 1, kernel_size=4, padding=1))
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, L, ab):
        img_input = torch.cat((L, ab), dim=1)
        return self.model(img_input)


if __name__ == "__main__":
    try:
        with open("hyperparameters.json", "r") as f:
            hparams = json.load(f)
        resolution = hparams.get("resolution", 256)
    except FileNotFoundError:
        resolution = 256
        print("Using default resolution: 256x256")

    generator = ViTUNetColorizer()
    generator_input_size = (1, 1, resolution, resolution)
    summary(generator, input_size=generator_input_size)
    
    discriminator = PatchDiscriminator()
    discriminator_input_size = [(1, 1, resolution, resolution), (1, 2, resolution, resolution)]
    summary(discriminator, input_size=discriminator_input_size)
