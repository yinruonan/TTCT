import os
from torch.utils.data import Dataset
from data.transforms import *
from glob import glob
import sys 
sys.path.append('../')
import logging
from utils.func import load_synthetic_sample, NormTensor, generate_attn_map, get_distance_weights
import cv2 as cv 
from matplotlib import pyplot as plt 


logger = logging.getLogger(__name__)

class Synthetic(Dataset):
    def __init__(self, root, n_slice=None, train=True, annot_err=False, norm=NormTensor(), cross_annot=False, sim_expt_ann=False, sigma_1=2.0, sigma_2=7.0):
        super().__init__()
        
        self.root = root
        self.n_slice = n_slice
        self.train = train
        self.annot_error = annot_err
        
        self.sim_expt_ann = sim_expt_ann
        
        self.cross_annot = cross_annot
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        
        self.err_slices = 0

        if train:
            seis_dir = os.path.join(root, 'train', 'seis')
            fault_dir = os.path.join(root, 'train', 'fault')
        else:
            seis_dir = os.path.join(root, 'test', 'seis')
            fault_dir = os.path.join(root, 'test', 'fault')
        
        self.seis_files = sorted(glob(os.path.join(seis_dir, '*.dat')))
        self.fault_files = sorted(glob(os.path.join(fault_dir, '*.dat')))
        
        # ==>> For data augmentation.
        _all = {'seismic', 'fault', 'attn_map', 'slice_weights'}
        _seis_only = {'seismic'}

        self.loc_aug = Compose([
            RandFlipd(_all, prob=0.2, spatial_axis=0),
            RandFlipd(_all, prob=0.2, spatial_axis=1),
            RandFlipd(_all, prob=0.2, spatial_axis=2),
            RandRotate90d(_all, prob=0.2, spatial_axes=(1, 2)),
            # RandScaleResizedCrop(_all, prob=0.3)
        ])
        self.int_aug = Compose([
            RandNoised(_seis_only, prob=0.2),
            # RandGammaTransfer(_seis_only, prob=0.2),
            # RandGaussianBlur(_seis_only, prob=0.2),
            # RandAdjustContrastd(_seis_only, prob=0.2, gamma=(0.8, 1.2)),
            # RandShiftIntensityd(_seis_only, prob=0.2, offsets=0.1)
        ])

        self.norm = norm
        
    
    def simulate_expert_annotation(self, fault_slice, min_area=200, min_length=50):

        binary = (fault_slice > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)
        weak_label = np.zeros_like(fault_slice, dtype=np.uint8)
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA]
            x, y, w, h = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], \
                        stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT]
            aspect_ratio = max(w / h, h / w)
            max_length = max(w, h)
            if area >= min_area and max_length >= min_length and aspect_ratio >= 0.1:
                weak_label[labels == i] = 1

        return weak_label
    
    
    def sparse_annot(self, fault):
        sparse_fault = np.zeros_like(fault) - 1.    # dhw
        attn_map = np.ones_like(fault).astype(np.float32)
        
        if self.cross_annot:
            annot_slice_ids = np.linspace(0, fault.shape[-1], num=self.n_slice // 2, endpoint=False, dtype=np.int16)
            sparse_fault[:, annot_slice_ids, :] = fault[:, annot_slice_ids, :]
            sparse_fault[:, :, annot_slice_ids] = fault[:, :, annot_slice_ids]
        else:
            annot_slice_ids = np.linspace(0, fault.shape[-1], num=self.n_slice, endpoint=False, dtype=np.int16)
            sparse_fault[:, annot_slice_ids, :] = fault[:, annot_slice_ids, :]
        
        slice_weights = get_distance_weights(annot_slice_ids, sigma=self.sigma_1)  # [1,h,1]
        
        for id in annot_slice_ids:
            if self.sim_expt_ann:
                ori_slice = sparse_fault[:, id, :].copy()
                sparse_fault[:, id, :] = self.simulate_expert_annotation(ori_slice, min_area=200, min_length=50)
                
                # if not np.array_equal(ori_slice, sparse_fault[:, id, :]):
                #     self.err_slices += 1
                #     if self.err_slices < 50:
                #         plt.imsave('./error_imgs/{}_ori.png'.format(str(self.err_slices)), ori_slice, cmap='jet')
                #         plt.imsave('./error_imgs/{}_exp.png'.format(str(self.err_slices)), sparse_fault[:, id, :], cmap='jet')
            
            slice = sparse_fault[:, id, :]
            slice_attn = generate_attn_map(slice, sigma=self.sigma_2)
            attn_map[:, id, :] = slice_attn
        
        return sparse_fault, attn_map, slice_weights
    
    def __getitem__(self, idx):

        seis_file = self.seis_files[idx]
        fault_file = self.fault_files[idx]

        # seis, fault: [d,h,w], numpy()
        seis, fault = load_synthetic_sample(seis_file, fault_file)
        if self.train and self.n_slice is not None:
            fault, attn_map, slice_weights = self.sparse_annot(fault)
        
        seis = torch.from_numpy(seis)[None]     # cdhw
        fault = torch.from_numpy(fault)[None]   # cdhw
        
        
        if not self.train:
            seis = self.norm(seis)
            return {'seis': seis, 
                    'fault': fault.squeeze(0),
                    'id': seis_file}
        
        attn_map = torch.from_numpy(attn_map)[None]   # cdhw
        slice_weights = torch.from_numpy(slice_weights)[None].expand(-1, 128, -1, 128)   # cdhw
        
        data = {'seismic': seis, 'fault': fault, 'attn_map': attn_map, 'slice_weights': slice_weights}
        data = self.loc_aug(data)
        data_weak = data.copy()
        self.loc_aug.shuffle()
        data = self.int_aug(data)
        self.int_aug.shuffle()
        try:
            seis = data['seismic'].as_tensor()
            fault = data['fault'].as_tensor()
            attn_map = data['attn_map'].as_tensor()
            slice_weights = data['slice_weights'].as_tensor()
        except:
            seis = data['seismic']
            fault = data['fault']
            attn_map = data['attn_map']
            slice_weights = data['slice_weights']
        
        try:
            seis_weak = data_weak['seismic'].as_tensor()
        except:
            seis_weak = data_weak['seismic']
        
        seis_weak = self.norm(seis_weak)
        seis = self.norm(seis)
        
        data = {
            'seis': seis, 
            'seis_weak': seis_weak,
            'fault': fault.squeeze(0),
            'attn_map': attn_map.squeeze(0),
            'slice_weights': slice_weights.squeeze(0),
            'id': seis_file
        }
        
        return data

    def __len__(self):
        return len(self.seis_files)


def get_dataset(root, n_slice=None, annot_err=False, cross_annot=False, sim_expt_ann=False, sigma_1=2.0, sigma_2=7.0):
     
    trainset = Synthetic(root, n_slice, train=True, annot_err=annot_err, cross_annot=cross_annot, sim_expt_ann=sim_expt_ann, sigma_1=sigma_1, sigma_2=sigma_2)
    validset = Synthetic(root, train=False)
    
    return {
        'trainset': trainset,
        'validset': validset
    }
    

class Equinor(Dataset):
    def __init__(self, nlb=8):
        super().__init__()
        seis_path = '/home/yinruonan/dataset/seismic/source/Equinor/Equinor.npy'
        fault_path = '/home/yinruonan/dataset/seismic/source/Equinor/Fault.npy'
        self.seismic = np.load(seis_path)
        fault =  np.load(fault_path)
        
        annot_slice_ids = np.linspace(0, fault.shape[1], num=nlb+2, endpoint=False, dtype=np.int16)[1:-1]
        self.sparse_fault = np.zeros_like(fault).astype(np.int64) - 1
        for idx in annot_slice_ids:
            self.sparse_fault[:, idx, :] = fault[:, idx, :]
        
        attn_map = np.ones_like(self.sparse_fault).astype(np.float32)
        
        slice_weights = get_distance_weights(annot_slice_ids, H=self.sparse_fault.shape[1], sigma=2.0)  # [1,h,1]
        
        for id in annot_slice_ids:
            slice = self.sparse_fault[:, id, :]
            slice_attn = generate_attn_map(slice, sigma=15)
            attn_map[:, id, :] = slice_attn
        
        attn_map = np.ones_like(self.sparse_fault).astype(np.float32)
        self.attn_map = torch.from_numpy(attn_map)  # dhw
        self.slice_weights = torch.from_numpy(slice_weights).expand(self.sparse_fault.shape[0], -1, self.sparse_fault.shape[2])   # dhw
        
        
        _all = {'seismic', 'fault', 'attn_map', 'slice_weights'}
        _seis_only = {'seismic'}
        
        self.loc_aug = Compose([
            RandFlipd(_all, prob=0.2, spatial_axis=0),
            RandFlipd(_all, prob=0.2, spatial_axis=1),
            RandFlipd(_all, prob=0.2, spatial_axis=2),
            RandRotate90d(_all, prob=0.2, spatial_axes=(1, 2)),
        ])
        self.int_aug = Compose([
            RandNoised(_seis_only, prob=0.2),
            # RandGammaTransfer(_seis_only, prob=0.2),
            # RandGaussianBlur(_seis_only, prob=0.2),
            # RandAdjustContrastd(_seis_only, prob=0.2, gamma=(0.8, 1.2)),
            # RandShiftIntensityd(_seis_only, prob=0.2, offsets=0.1)
        ])

        self.norm = NormTensor()
    
    def __getitem__(self, idx):
        h, w, d = self.seismic.shape
    
        rand_h = random.randint(0, h-128)
        rand_w = random.randint(0, w-128)
        rand_d = random.randint(0, d-128)
        
        seismic = self.seismic[rand_h:rand_h+128, rand_w:rand_w+128, rand_d:rand_d+128]
        fault = self.sparse_fault[rand_h:rand_h+128, rand_w:rand_w+128, rand_d:rand_d+128]
        attn_map = self.attn_map[rand_h:rand_h+128, rand_w:rand_w+128, rand_d:rand_d+128]
        slice_weights = self.slice_weights[rand_h:rand_h+128, rand_w:rand_w+128, rand_d:rand_d+128]
        
        
        seismic = torch.from_numpy(seismic)[None]
        fault = torch.from_numpy(fault)[None]
        attn_map = attn_map[None]
        slice_weights = slice_weights[None]
        
        data = {'seismic': seismic, 'fault': fault, 'attn_map': attn_map, 'slice_weights': slice_weights}
        weak_aug_data = self.loc_aug(data)
        self.loc_aug.shuffle()
        strong_aug_data = self.int_aug(weak_aug_data)
        self.int_aug.shuffle()
        seis_weak,  seis_strong, fault = weak_aug_data['seismic'].as_tensor(), \
            strong_aug_data['seismic'].as_tensor(), weak_aug_data['fault'].as_tensor()
        attn_map = weak_aug_data['attn_map'].as_tensor()
        slice_weights = weak_aug_data['slice_weights'].as_tensor()
        
        seis_weak = self.norm(seis_weak)
        seis_strong = self.norm(seis_strong)
        
        data = {
            'seis': seis_strong, 
            'seis_weak': seis_weak,
            'fault': fault.squeeze(0),
            'attn_map': attn_map.squeeze(0),
            'slice_weights': slice_weights.squeeze(0),
        }
        return data

        
    def __len__(self):
        return 200


class Thebe(Dataset):
    def __init__(self, nlb=8):
        super().__init__()
        filepath = '/home/yinruonan/dataset/seismic/TheBe_NPY/Thebe.npz'
        thebe = np.load(filepath)
        self.seismic = thebe['seismic'].transpose(2, 0, 1).astype(np.float32)
        fault = thebe['fault'].transpose(2, 0, 1).astype(np.int64)
        
        annot_slice_ids = np.linspace(0, fault.shape[1], num=nlb, endpoint=False, dtype=np.int16)
        self.sparse_fault = np.zeros_like(fault).astype(np.int64) - 1
        for idx in annot_slice_ids:
            self.sparse_fault[:, idx, :] = fault[:, idx, :]
        
        attn_map = np.ones_like(self.sparse_fault).astype(np.float32)
        
        slice_weights = get_distance_weights(annot_slice_ids, H=self.sparse_fault.shape[1], sigma=2.0)  # [1,h,1]
        
        for id in annot_slice_ids:
            slice = self.sparse_fault[:, id, :]
            slice_attn = generate_attn_map(slice, sigma=20)
            attn_map[:, id, :] = slice_attn
        
        self.attn_map = torch.from_numpy(attn_map)  # dhw
        self.slice_weights = torch.from_numpy(slice_weights).expand(self.sparse_fault.shape[0], -1, self.sparse_fault.shape[2])   # dhw
        
        
        _all = {'seismic', 'fault', 'attn_map', 'slice_weights'}
        _seis_only = {'seismic'}
        
        self.loc_aug = Compose([
            RandFlipd(_all, prob=0.2, spatial_axis=0),
            RandFlipd(_all, prob=0.2, spatial_axis=1),
            RandFlipd(_all, prob=0.2, spatial_axis=2),
            RandRotate90d(_all, prob=0.2, spatial_axes=(1, 2)),
        ])
        self.int_aug = Compose([
            RandNoised(_seis_only, prob=0.2),
            # RandGammaTransfer(_seis_only, prob=0.2),
            # RandGaussianBlur(_seis_only, prob=0.2),
            # RandAdjustContrastd(_seis_only, prob=0.2, gamma=(0.8, 1.2)),
            # RandShiftIntensityd(_seis_only, prob=0.2, offsets=0.1)
        ])

        self.norm = NormTensor()
    
    def __getitem__(self, idx):
        h, w, d = self.seismic.shape
        rand_h = random.randint(0, h-128)
        rand_w = random.randint(0, w-128)
        rand_d = random.randint(0, d-128)
        
        seismic = self.seismic[rand_h:rand_h+128, rand_w:rand_w+128, rand_d:rand_d+128]
        fault = self.sparse_fault[rand_h:rand_h+128, rand_w:rand_w+128, rand_d:rand_d+128]
        attn_map = self.attn_map[rand_h:rand_h+128, rand_w:rand_w+128, rand_d:rand_d+128]
        slice_weights = self.slice_weights[rand_h:rand_h+128, rand_w:rand_w+128, rand_d:rand_d+128]
        
        seismic = torch.from_numpy(seismic)[None]
        fault = torch.from_numpy(fault)[None]
        attn_map = attn_map[None]
        slice_weights = slice_weights[None]
        
        data = {'seismic': seismic, 'fault': fault, 'attn_map': attn_map, 'slice_weights': slice_weights}
        weak_aug_data = self.loc_aug(data)
        self.loc_aug.shuffle()
        strong_aug_data = self.int_aug(weak_aug_data)
        self.int_aug.shuffle()
        seis_weak,  seis_strong, fault = weak_aug_data['seismic'].as_tensor(), \
            strong_aug_data['seismic'].as_tensor(), weak_aug_data['fault'].as_tensor()
        attn_map = weak_aug_data['attn_map'].as_tensor()
        slice_weights = weak_aug_data['slice_weights'].as_tensor()
        
        seis_weak = self.norm(seis_weak)
        seis_strong = self.norm(seis_strong)
        
        data = {
            'seis': seis_strong, 
            'seis_weak': seis_weak,
            'fault': fault.squeeze(0),
            'attn_map': attn_map.squeeze(0),
            'slice_weights': slice_weights.squeeze(0),
        }
        return data

        
    def __len__(self):
        return 200
