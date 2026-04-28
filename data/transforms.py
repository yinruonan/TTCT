import torch.nn.functional
import numpy as np
import torch
from torchvision.transforms import functional as F
import random
from monai.transforms import RandFlipd, RandRotate90d, RandGaussianNoised, RandAdjustContrastd, \
    RandGaussianSmoothd, RandSpatialCropd, SpatialPadd, RandRotated, RandGaussianSharpend, \
    RandHistogramShiftd, ToTensord, RandShiftIntensityd


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def shuffle(self):
        random.shuffle(self.transforms)

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class RandNoised:
    def __init__(self, keys, prob=0.1):
        
        self.prob = prob
        self.keys = keys
    
    def __call__(self, data):
        
        if np.random.uniform() < self.prob:
            s_max, s_min = torch.max(data['seismic']), torch.min(data['seismic'])
            scale = random.uniform(0.1, 1.0) * (s_max - s_min) * 0.16
            data = RandGaussianNoised(self.keys, prob=1, std=scale)(data)
            
        return data

class RandZoomPadd:
    def __init__(self, keys, scale=(0.8, 0.95), prob=0.1):
        
        assert scale[0] <= 1 and scale[1] <= 1 
        self.prob = prob
        self.scale = scale
        self.keys = keys
        self.spatial_pad = SpatialPadd(keys=self.keys, spatial_size=(128, 128, 128), mode='reflect')
    
    def __call__(self, data):
        
        if np.random.uniform() < self.prob:
            # resize 
            new_size = [int(np.random.uniform(low=self.scale[0], high=self.scale[1]) * 128) for _ in range(3)]
            for key in self.keys:
                
                mode = 'trilinear'
                align_corners = True
                if key == 'fault':
                    mode = 'nearest'
                    align_corners = None
                val = torch.nn.functional.interpolate(data[key].unsqueeze(0), size=new_size, mode=mode, align_corners=align_corners)
            
                # update results
                data.update({key: val.squeeze(0)})
            data = self.spatial_pad(data)
        return data


class RandScaleResizedCrop:
    def __init__(self, keys, prob=0.1):

        self.prob = prob
        self.keys = keys
        self.spatial_crop = RandSpatialCropd(
            keys, roi_size=(128, 128, 128), random_size=False)

    def __call__(self, data):

        if np.random.uniform() < self.prob:
            # resize
            new_tl = int(np.random.uniform(1.5, high=2)) * 128
            new_il = int(np.random.uniform(1, high=1.2)) * 128
            new_xl = int(np.random.uniform(1, high=1.2)) * 128
            new_size = [new_tl, new_il, new_xl]
            for key in self.keys:
                mode = 'trilinear'
                align_corners = True
                if key == 'fault':
                    mode = 'nearest'
                    align_corners = None
                val = torch.nn.functional.interpolate(data[key].unsqueeze(0), size=new_size, mode=mode, align_corners=align_corners)
            # update results
                data.update({key: val.squeeze(0)})
            data = self.spatial_crop(data)

        return data


class RandGammaTransfer:
    def __init__(self, keys, prob=0.1):
        
        self.prob = prob
        self.keys = keys
    
    def __call__(self, data):
        
        if np.random.uniform() < self.prob:
            s_max, s_min = torch.max(data['seismic']), torch.min(data['seismic'])
            if random.randint(0, 1):
                gamma = random.uniform(0.6667, 1)
            else:
                gamma = random.uniform(1, 1.5)
            gamma_seismic = (data['seismic'] - s_min) ** gamma

            gamma_range = torch.max(gamma_seismic) - torch.min(gamma_seismic)
            gamma_seismic = ((gamma_seismic - torch.min(gamma_seismic)) / gamma_range) * (s_max - s_min) + s_min
            data.update({'seismic': gamma_seismic})
            
        return data



class RandGaussianBlur:
    
    def __init__(self, keys, sigma=(0.1, 0.7), prob=0.1):
        
        self.prob = prob
        self.sigma = sigma
        self.keys = keys 
        
        self.blur_func = [self.gaussian_blur_t, self.gaussian_blur_h, self.gaussian_blur_w]
    
    def gaussian_blur_t(self, seismic):
        if np.random.uniform() < 0.5:
            sigma_1 = random.uniform(self.sigma[0], self.sigma[1]) 
            sigma_2 = random.uniform(self.sigma[0], self.sigma[1])
            return F.gaussian_blur(seismic, kernel_size=[3, 3], sigma=[sigma_1, sigma_2])
        return seismic
    
    def gaussian_blur_w(self, seismic):
        if np.random.uniform() < 0.5:
            sigma_1 = random.uniform(self.sigma[0], self.sigma[1]) 
            sigma_2 = random.uniform(self.sigma[0], self.sigma[1])
            seismic = seismic.permute(0, 3, 2, 1)
            seismic = F.gaussian_blur(seismic, kernel_size=[3, 3], sigma=[sigma_1, sigma_2])
            seismic = seismic.permute(0, 3, 2, 1)
        return seismic
    
    def gaussian_blur_h(self, seismic):
        if np.random.uniform() < 0.5:
            sigma_1 = random.uniform(self.sigma[0], self.sigma[1]) 
            sigma_2 = random.uniform(self.sigma[0], self.sigma[1])
            seismic = seismic.permute(0, 2, 1, 3)
            seismic = F.gaussian_blur(seismic, kernel_size=[3, 3], sigma=[sigma_1, sigma_2])
            seismic = seismic.permute(0, 2, 1, 3)
        return seismic
    
    def __call__(self, data):
        if np.random.uniform() < self.prob:
            seismic = data['seismic']
            random.shuffle(self.blur_func)
            
            for blur_func in self.blur_func:
                seismic = blur_func(seismic)

            data.update({'seismic': seismic})
        
        return data 