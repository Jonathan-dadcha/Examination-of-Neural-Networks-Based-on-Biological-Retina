import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FoveatedConv2d(nn.Module):
    """
    שכבת קונבולוציה פוביאלית (Foveated Layer).
    מממשת את עקרון ה-Location-dependent processing:
    1. Foveal Stream: עיבוד מלא ברזולוציה גבוהה.
    2. Peripheral Stream: עיבוד חסכוני ברזולוציה נמוכה.
    3. Blending: איחוד מבוסס מרחק מהמרכז (Gaussian Mask).
    """
    def __init__(self, in_channels, out_channels, kernel_size, img_size, fovea_radius=0.5):
        super(FoveatedConv2d, self).__init__()
        
        # הנתיב הפוביאלי (המרכז החד)
        self.conv_fovea = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        
        # הנתיב הפריפריאלי (ההיקף) - יכולנו להשתמש באותם משקולות, אבל הפרדה מאפשרת התמחות
        self.conv_periph = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        
        # יצירת מסכת פוביאה קבועה (Gaussian Mask)
        self.img_size = img_size
        self.register_buffer('mask', self._create_gaussian_mask(img_size, sigma=fovea_radius))

    def _create_gaussian_mask(self, size, sigma):
        """יוצר מפת חום שבה המרכז הוא 1 והצדדים דועכים ל-0"""
        H, W = size
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        # פונקציית גאוס: e^(-x^2 / 2sigma^2)
        mask = torch.exp(-(dist**2) / (2 * sigma**2))
        return mask.view(1, 1, H, W)

    def forward(self, x):
        # 1. נתיב פוביאלי: קונבולוציה רגילה על הקלט המקורי
        out_fovea = self.conv_fovea(x)
        
        # 2. נתיב פריפריאלי: "ראייה מטושטשת" וחסכונית
        # אנחנו מדמים את זה ע"י הקטנת הקלט (Downsample) לחצי, הפעלת קונבולוציה, והגדלה חזרה.
        # זה מדמה אובדן מידע בפריפריה.
        x_low_res = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        out_periph_low = self.conv_periph(x_low_res)
        out_periph = F.interpolate(out_periph_low, size=out_fovea.shape[2:], mode='bilinear', align_corners=False)
        
        # התאמת המסכה לגודל הפלט (כי הקונבולוציה ללא padding מקטינה את התמונה)
        if self.mask.shape[2:] != out_fovea.shape[2:]:
            current_mask = F.interpolate(self.mask, size=out_fovea.shape[2:], mode='bilinear', align_corners=False)
        else:
            current_mask = self.mask

        # 3. איחוד משוקלל (Soft Blending)
        # המרכז מקבל בעיקר מה-Fovea, הצדדים מה-Periphery
        out = out_fovea * current_mask + out_periph * (1 - current_mask)
        
        return out