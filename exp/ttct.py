import os
import torch
import utils.func as func
from torch.amp.grad_scaler import GradScaler
from model.vnet import VNet
from model.unet import UNet2D
import logging
from einops import rearrange

logger = logging.getLogger(__name__)


class TTCT:
    
    def __init__(self, args, criterion, device, writer=None):
        
        self.args = args
        self.writer = writer
        self.it = 0
        self.max_it = 0
        
        self.grad_scaler = GradScaler()
        self.gpu = device

        self.model3d = VNet().to(self.gpu)
        self.model2d = UNet2D().to(self.gpu)
        
        self.model3d_opt = torch.optim.Adam(self.model3d.parameters(), lr=args.lr)
        self.model2d_opt = torch.optim.Adam(self.model2d.parameters(), lr=args.lr)
       
        self.seg_criterion = criterion
        
        self.iter_loss = {}
        
        self.avg_loss = {
            'l_seg3d': func.AverageMeter(),
            'l_seg2d': func.AverageMeter()
        }
        
    def set_scheduler(self):
        self.model3d_lr_sch = func.PolyLR(
            self.model3d_opt, self.it, max_it=self.max_it, power=self.args.power)
        
        self.model2d_lr_sch = func.PolyLR(
            self.model2d_opt, self.it, max_it=self.max_it, power=self.args.power)

    def update_lr(self):
        self.model3d_lr_sch.step()
        self.model2d_lr_sch.step()
    
    def load_ckpts(self, ckpt):
        state_dict = torch.load(ckpt, weights_only=True)
        self.model3d.load_state_dict(state_dict['model3d'])
        self.model2d.load_state_dict(state_dict['model2d'])
    
    
    def train(self):
        self.model3d.train()
        self.model2d.train()
        
    def eval(self):
        self.model3d.eval()
        self.model2d.eval()
    
    def train_iter(self, data):
        img = data['seis'].to(self.gpu)
        fault = data['fault'].long().to(self.gpu)   # [b,d,h,w], labeled at dimension 'h'
        attn_map = data['attn_map'].to(self.gpu)
        if self.args.sim_expt_ann:
            attn_map[attn_map >= 0.01] = 1
            attn_map[attn_map < 0.01] = 0
        else:
            attn_map[attn_map >= 0] = 1 # without attn_map, set all weights to 1.
            
        slice_weights = data['slice_weights'].to(self.gpu)
        
        self.train()
        self.model3d_opt.zero_grad()
        self.model2d_opt.zero_grad()
        
        _lambda = func.sigmoid_rampup(self.it, self.args.rampup)
        
        with torch.amp.autocast(device_type='cuda'):
            # =====>>>  Forward 3D model   <<<===== # 
            logits_3d = self.model3d(img)['logits']
            conf_3d, hard_3d = torch.max(torch.softmax(logits_3d, dim=1), dim=1)

            # =====>>>  Forward 2D model   <<<===== # 
            img_2d = rearrange(img, 'b c d h w -> (b h) c d w') # b*h,c,d,w
            img_2d = func.zscore_norm(img_2d)
            logits_2d = self.model2d(img_2d)['logits']    # [b*h,2,d,w]
            logits_2d = rearrange(logits_2d, '(b h) c d w -> b c d h w', h=128)
            conf_2d, hard_2d = torch.max(torch.softmax(logits_2d, dim=1), dim=1)    # [b,d,h,w]
            
            # =====>>> Mix Label & Loss for 3D model <<<===== # 
            pseudo_for_3d = torch.where(fault == -1, hard_2d, fault)
            slice_weights = max(1. - _lambda, 0.) * slice_weights    # ramp_down
            slice_weights[fault != -1] = 1.
            l_seg3d = self.seg_criterion(logits_3d, pseudo_for_3d, weights=slice_weights * attn_map)
            
            # =====>>> Mix Label & Loss for 2D model <<<===== # 
            pseudo_for_2d = torch.where(fault == -1, hard_3d, fault)
            weights_2d = conf_3d > conf_2d
            weights_2d = _lambda * weights_2d
            weights_2d[fault != -1] = 1.
            l_seg2d = self.seg_criterion(logits_2d, pseudo_for_2d, weights=weights_2d * attn_map)
            
            l_total = l_seg3d + l_seg2d
            
        self.iter_loss.update({'l_seg3d': l_seg3d.detach().cpu().item()})
        self.iter_loss.update({'l_seg2d': l_seg2d.detach().cpu().item()})
        
        self.grad_scaler.scale(l_total).backward()
        self.grad_scaler.step(self.model3d_opt)
        self.grad_scaler.step(self.model2d_opt)
        self.grad_scaler.update()
        
        # iter completed.
        self.it += 1
        self.moving_avg_loss()
        
    def moving_avg_loss(self):
        for k, v in self.iter_loss.items():
            self.iter_loss.update({k: v})
            self.avg_loss[k].update(self.iter_loss[k])
    
    def reset_avg_loss(self):
        for _, v in self.avg_loss.items():
            v.reset()
    
    def get_seg_model(self):
        return self.model3d

    def get_seg_model2d(self):
        return self.model2d
        
    def get_t_seg_model(self):
        return None
    
    def save_ckpt(self, fp):
        save_dir = os.path.join(self.args.ckpts_dir, func.get_exp_dir_name(self.args))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        state_dict = {
            'model3d': self.model3d.state_dict(),
            'model2d': self.model2d.state_dict(),
        }

        torch.save(state_dict, os.path.join(save_dir, fp))
        
    def log2board(self, writer, tag='it', epoch=0):
        assert tag in ['it', 'avg']
        loss_str = []
        if tag == 'it':
            for k, v in self.iter_loss.items():
                writer.add_scalar(os.path.join(tag, k), v, self.it)
                loss_str.append('{}: {:.3f}'.format(k, v))
        else:
            for k, v in self.avg_loss.items():
                writer.add_scalar(os.path.join(tag, k), v.avg, epoch)
                loss_str.append('{}: {:.3f}'.format(k, v.avg))

            self.reset_avg_loss()

        loss_str = ' '.join(loss_str)
        return loss_str