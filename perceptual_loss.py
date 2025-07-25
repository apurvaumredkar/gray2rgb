import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22], loss_type='l2', resize=True):

        super().__init__()
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = layers
        self.loss_type = loss_type
        self.resize = resize

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _preprocess(self, img):
    
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)  
        if self.resize:
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img = (img - self.mean) / self.std
        return img

    def forward(self, input, target):
        input = self._preprocess(input)
        target = self._preprocess(target)

        feats_input = []
        feats_target = []
        x = input
        y = target
        for i, layer in enumerate(self.vgg):  # pyright: ignore[reportArgumentType]
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                feats_input.append(x)
                feats_target.append(y)

        loss = 0.0
        for f_in, f_tar in zip(feats_input, feats_target):
            if self.loss_type == 'l1':
                loss += F.l1_loss(f_in, f_tar)
            else:
                loss += F.mse_loss(f_in, f_tar)
        return loss
