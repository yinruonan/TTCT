import os
import torch
import utils.func as func
from torch.amp.grad_scaler import GradScaler
from model.vnet import VNet

class Baseline:
    
    def __init__(self, args, criterion, device, writer=None):
        
        self.args = args
        
        self.it = 0
        self.max_it = 0
        
        self.writer = writer
        
        self.grad_scaler = GradScaler()
        self.gpu = device
        
        self.model3d = VNet().to(self.gpu)
        self.t_model = func.Teacher(self.model3d)
        
        self.model3d_opt = torch.optim.Adam(self.model3d.parameters(), lr=args.lr)
       
        self.seg_criterion = criterion
        
        self.iter_loss = {}
        
        self.avg_loss = {
            'l_seg3d': func.AverageMeter(),
        }
        
    def set_scheduler(self):
        self.model3d_lr_sch = func.PolyLR(
            self.model3d_opt, self.it, max_it=self.max_it, power=self.args.power)
        
    def update_lr(self):
        self.model3d_lr_sch.step()
    
    def load_ckpts(self, ckpt):
        state_dict = torch.load(ckpt, weights_only=True)
        self.model3d.load_state_dict(state_dict['model3d'])
    
    
    def train(self):
        self.model3d.train()
        
    def eval(self):
        self.model3d.eval()
    
    def train_iter(self, data):
        img = data['seis'].to(self.gpu)
        fault = data['fault'].to(self.gpu)
        
        self.train()
        self.model3d_opt.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):
            logits = self.model3d(img)['logits']

            l_seg3d = self.seg_criterion(logits, fault)
            
            
        self.iter_loss.update({'l_seg3d': l_seg3d.detach().cpu().item()})
        
        self.grad_scaler.scale(l_seg3d).backward()
        self.grad_scaler.step(self.model3d_opt)
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
        return None 
    
    def save_ckpt(self, fp):
        save_dir = os.path.join(self.args.ckpts_dir, func.get_exp_dir_name(self.args))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        state_dict = {
            'model3d': self.model3d.state_dict(),
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