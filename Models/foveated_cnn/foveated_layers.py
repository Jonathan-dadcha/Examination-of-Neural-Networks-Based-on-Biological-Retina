import torch
import torch.nn as nn
import torch.nn.functional as F

class FoveatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, img_size, fovea_radius=0.5):
        super(FoveatedConv2d, self).__init__()
        
        self.conv_fovea = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        self.conv_periph = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        self.img_size = img_size
        self.register_buffer('mask', self._create_gaussian_mask(img_size, sigma=fovea_radius))

    def _create_gaussian_mask(self, size, sigma):
        """Create a heatmap where center is 1 and edges decay to 0."""
        H, W = size
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        mask = torch.exp(-(dist**2) / (2 * sigma**2))
        return mask.view(1, 1, H, W)

    def forward(self, x):
        out_fovea = self.conv_fovea(x)

        x_low_res = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        out_periph_low = self.conv_periph(x_low_res)
        out_periph = F.interpolate(out_periph_low, size=out_fovea.shape[2:], mode='bilinear', align_corners=False)
        
        if self.mask.shape[2:] != out_fovea.shape[2:]:
            current_mask = F.interpolate(self.mask, size=out_fovea.shape[2:], mode='bilinear', align_corners=False)
        else:
            current_mask = self.mask

        out = out_fovea * current_mask + out_periph * (1 - current_mask)
        
        return out