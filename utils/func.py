import torch
import os
import numpy as np
from einops import rearrange
import datetime
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import random
import torch.nn as nn
import copy
from utils.metric import MMetric
import cfg 
from tqdm import tqdm
import math 
from scipy.ndimage import distance_transform_edt
import logging 

logger = logging.getLogger(__name__)


def load_synthetic_sample(seis_file, fault_file):
    seis = np.fromfile(seis_file, dtype=np.single).reshape(128, 128, 128)   # [h,w,d] float32
    fault = np.fromfile(fault_file, dtype=np.single).reshape(128, 128, 128)  # [h,w,d] float32
    # to [d,h,w]
    seis = np.transpose(seis).astype(np.float32)
    fault = np.transpose(fault).astype(np.float32)
    return seis, fault


class NormTensor:
    def __init__(self):
        pass
    
    def __call__(self, x):
        x_mean = torch.mean(x)
        x_std = torch.std(x)

        x = (x - x_mean) / (x_std + 1e-7)
        return x 
        
        
class NormNpy:
    def __init__(self):
        pass
    
    def __call__(self, x):
        x_mean = np.mean(x)
        x_std = np.std(x)
        x = (x - x_mean) / (x_std + 1e-7)
        return x 


class Teacher:
    def __init__(self, segmentor):
        self.segmentor = copy.deepcopy(segmentor)
        for param in self.segmentor.parameters():
            param.detach_()
            param.requires_grad = False

    @torch.no_grad()
    def ema_update(self, segmentor, it, max_it):

        alpha = 1 - (1 - 0.995) * (math.cos(math.pi * it / max_it) + 1) / 2

        for param, t_param in zip(segmentor.parameters(), self.segmentor.parameters()):
            t_param.data = t_param.data * alpha + param.data * (1. - alpha)

        for (s_name, s_module), (t_name, t_module) in zip(segmentor.named_modules(), self.segmentor.named_modules()):
            if isinstance(s_module, torch.nn.BatchNorm3d) and isinstance(t_module, torch.nn.BatchNorm3d):
                if s_name == t_name:
                    t_module.running_mean = t_module.running_mean.clone().detach() * alpha +\
                        s_module.running_mean.clone().detach() * (1. - alpha)
                    t_module.running_var = t_module.running_var.clone().detach() * alpha +\
                        s_module.running_var.clone().detach() * (1. - alpha)
    
    def train(self):
        self.segmentor.train()
        
    def eval(self):
        self.segmentor.eval()
    
    def __call__(self, x, **kwargs):
        with torch.no_grad():
            return self.segmentor(x, **kwargs) 

def minmax_norm(data):
    
    def _norm(data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    
    b = data.size(0)
    result = []
    for i in range(b):
        result.append(_norm(data[i])[None])
    return torch.cat(result, dim=0)


def zscore_norm(data):
    
    def _norm(data):
        return (data - torch.mean(data)) / (torch.std(data) + 1e-10)
    
    b = data.size(0)
    result = []
    for i in range(b):
        result.append(_norm(data[i])[None])
    return torch.cat(result, dim=0)


def get_exp_dir_name(args):

    name = '{}-{}'.format(args.alg, args.n_slice)
    if args.exp_name != '':
        name += '-{}'.format(args.exp_name)
    
    return name 

def get_writer(args):

    exp_dir_name = get_exp_dir_name(args)
    logdir = os.path.join(os.getcwd(), args.log_dir, exp_dir_name)

    tb_writer = SummaryWriter(log_dir=logdir, flush_secs=30)

    desc = []
    for name, val in sorted(vars(args).items()):
        desc.append('{}: {}'.format(str(name), str(val)))

    tb_writer.add_text('Hyperparameters', '  \n'.join(desc), global_step=0)

    return tb_writer


def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PolyLR:

    def __init__(self, optimizer, it, max_it, power=0.9):
        self.optimizer = optimizer
        self.lr = [group['lr'] for group in self.optimizer.param_groups]
        self.it = it
        self.max_it = max_it
        self.power = power

    def lr_poly(self, lr):
        return lr * ((1 - float(self.it) / self.max_it) ** self.power)

    def step(self):
        self.it += 1
        lr = [self.lr_poly(lr) for lr in self.lr]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]


class AverageMeter:

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _pad(img, infer_size):
    _, _, t, h, w = img.size()
    pad_t = t % infer_size[0]
    pad_h = h % infer_size[1]
    pad_w = w % infer_size[2]
    pad3d = nn.ReflectionPad3d(padding=(0, infer_size[2] - pad_w if pad_w != 0 else 0,
                                        0, infer_size[1] - pad_h if pad_h != 0 else 0,
                                        0, infer_size[0] - pad_t if pad_t != 0 else 0))

    new_img = pad3d(img)

    return new_img


def model_infer(model, img, infer_size, device, norm=NormTensor()):
    img = torch.from_numpy(img)[None, None]
    
    if infer_size is None:
        img = img.to(device)
        with torch.no_grad() and torch.amp.autocast('cuda'):
            img = norm(img)
            pred = model(img, turnoff_drop=True)['logits']
            
        pred = torch.softmax(pred, dim=1)
        pred = pred[0, 1, ...].detach().cpu().numpy()
        return pred

    if isinstance(infer_size, int):
        infer_size = (infer_size, infer_size, infer_size)
    
    _, _, ori_d, ori_h, ori_w = img.shape
    img = _pad(img, infer_size)
    
    result = np.zeros(img.shape, np.single)
    _, _, D, H, W = img.shape

    ds = [idx for idx in range(0, D+1, infer_size[0]) if idx + infer_size[0] <= D]
    hs = [idx for idx in range(0, H+1, infer_size[1]) if idx + infer_size[1] <= H]
    ws = [idx for idx in range(0, W+1, infer_size[2]) if idx + infer_size[2] <= W]
    
    for d in ds:
        for h in hs:
            for w in ws:
                cube = img[:, :, d:d+infer_size[0],
                                h:h+infer_size[1], w:w+infer_size[2]]
                cube = norm(cube).to(device)
                with torch.amp.autocast('cuda') and torch.no_grad():
                    pred = model(cube, turnoff_drop=True)['logits']
                    
                pred = torch.softmax(pred, dim=1)
                pred = pred[:, 1, ...].detach().cpu().numpy()
                result[:, 0, d:d+infer_size[0], h:h+infer_size[1], w:w+infer_size[2]] = pred
    return result[0, 0, :ori_d, :ori_h, :ori_w]


def model_infer_2d(model, img, infer_size, device, norm=NormTensor()):
    results = np.zeros(img.shape, dtype=np.float32) # dhw
    img = torch.from_numpy(img)[None, None] # [1,1,d,h,w]
    
    _, _, ori_d, ori_h, ori_w = img.shape
    
    img = _pad(img, infer_size=(16, 16, 16))[:, :, :, :ori_h, :]
    img = rearrange(img, 'b c d h w -> (b h) c d w')
    img = img.to(device)
    
    for idx in range(img.shape[0]):
        cur_img = norm(img[idx])[None]
        with torch.no_grad() and torch.amp.autocast('cuda'):
            pred = model(cur_img, turnoff_drop=True)['logits']
        pred = torch.softmax(pred, dim=1)
        pred = pred[0, 1, ...].detach().cpu().numpy()[:ori_d, :ori_w]
        
        results[:, idx, :] = pred
    
    return results



def prob2board(seismic, results, slice_id, writer, tag, global_step, remain_prob=True):
    
    if not remain_prob:
        results[results >= 0.5] = 1.
        results[results < 0.5] = 0.
    
    
    def _norm(x):
        x_mean = np.mean(x)
        x_std = np.std(x)
        x = (x - x_mean) / (x_std + 1e-7)
        x = np.clip(x, a_min=-3.2, a_max=3.2)
        x_min = np.min(x)
        x_max = np.max(x)
        return (x - x_min) / (x_max - x_min)

    seismic = _norm(seismic)
    # tline result
    seis_slice_tline = plt.get_cmap('gray')(seismic[slice_id['tline'], :, :])
    pred_slice_tline = (results[slice_id['tline'], :, :, None] >= 0.5).astype(np.int8)
    foregrnd_tline = plt.get_cmap('jet')(results[slice_id['tline'], :, :])
    prob_on_seis = np.where(pred_slice_tline, foregrnd_tline, seis_slice_tline)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(prob_on_seis)
    writer.add_figure(tag + ' tline', fig, global_step=global_step)
    
    # inline 
    seis_slice_iline = plt.get_cmap('gray')(seismic[:, slice_id['iline'], :])
    pred_slice_iline = (results[:, slice_id['iline'], :, None] >= 0.5).astype(np.int8)
    foregrnd_iline = plt.get_cmap('jet')(results[:, slice_id['iline'], :])
    prob_on_seis = np.where(pred_slice_iline, foregrnd_iline, seis_slice_iline)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(prob_on_seis)
    writer.add_figure(tag + ' iline', fig, global_step=global_step)
    
    # xline
    seis_slice_xline = plt.get_cmap('gray')(seismic[:, :, slice_id['xline']])
    pred_slice_xline = (results[:, :, slice_id['xline'], None] >= 0.5).astype(np.int8)
    foregrnd_xline = plt.get_cmap('jet')(results[:, :, slice_id['xline']])
    prob_on_seis = np.where(pred_slice_xline, foregrnd_xline, seis_slice_xline)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(prob_on_seis)
    writer.add_figure(tag + ' xline', fig, global_step=global_step)


def compute_entropy(logits):
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -torch.sum(probs * log_probs, dim=1)  # shape: [B, D, H, W] or [B, H, W]
    return entropy


def cal_metric2board(results, test_data_label, writer, it, tag=''):
    from utils.metric import MMetric
    
    metrics = MMetric()
            
    results[results >= 0.5] = 1
    results[results < 0.5] = 0
    metrics.update(results, test_data_label)
    fmt = '[Epoch {} Evaluate on test data] | IoU {:.3f}, Precision {:.3f}, Dice {:.3f}, Recall {:.3f}, F1Score {:.3f}'
    logger.info(fmt.format(it + 1, metrics.iou(), metrics.precision(), metrics.dice(), metrics.recall(), metrics.f1score()))
    
    writer.add_scalar(os.path.join(
        'Evaluate on test data', tag, 'IoU'), metrics.iou(), it)
    writer.add_scalar(os.path.join(
        'Evaluate on test data', tag, 'Precision'), metrics.precision(), it)
    writer.add_scalar(os.path.join(
        'Evaluate on test data', tag, 'Dice'), metrics.dice(), it)
    writer.add_scalar(os.path.join(
        'Evaluate on test data', tag, 'Recall'), metrics.recall(), it)
    writer.add_scalar(os.path.join(
        'Evaluate on test data', tag, 'F1Score'), metrics.f1score(), it)


def evaluate(model, val_loader, writer, epoch, device, is_3d=True):
    model.eval()
    metrics = MMetric()

    val_loader = tqdm(val_loader)
    
    with torch.amp.autocast(device_type='cuda'):
        Ps = []
        GTs = []
        for idx, data in enumerate(val_loader):
            seismic = data['seis'].to(device)
            fault = data['fault']
            
            if not is_3d:
                seismic = rearrange(seismic, 'b c d h w -> (b h) c d w')
                for i in range(seismic.size(0)):
                    temp_img = seismic[i]
                    temp_img = (temp_img - temp_img.mean()) / (temp_img.std() + 1e-7)
                    seismic[i] = temp_img
            
            with torch.no_grad():
                pred = model(seismic, turnoff_drop=True)['logits']
                if not is_3d:
                    pred = rearrange(pred, '(b h) c d w -> b c d h w', h=128)
                
            pred = torch.argmax(torch.softmax(pred.detach(), dim=1), dim=1, keepdim=True)  # [b,1,d,h,w]
            pred = pred.cpu().numpy()[:, 0]

            for j in range(pred.shape[0]):
                Ps.append(pred[j])
                GTs.append(fault.numpy()[j])

        Ps = np.concatenate(Ps, axis=2)
        GTs = np.concatenate(GTs, axis=2)
        metrics.update(Ps, GTs)
    
    # metrics to tensorboard
    fmt = '[Epoch {} Evaluate {}] | IoU {:.3f}, Precision {:.3f}, Dice {:.3f}, Recall {:.3f}, F1Score {:.3f}'
    logger.info(fmt.format(epoch + 1, '3D' if is_3d else '2D', metrics.iou(), metrics.precision(), metrics.dice(), metrics.recall(), metrics.f1score()))
    
    eval_mode = 'evaluate3D' if is_3d else 'evaluate2D'
    writer.add_scalar(os.path.join(
        eval_mode, 'IoU'), metrics.iou(), epoch)
    writer.add_scalar(os.path.join(
        eval_mode, 'Precision'), metrics.precision(), epoch)
    writer.add_scalar(os.path.join(
        eval_mode, 'Dice'), metrics.dice(), epoch)
    writer.add_scalar(os.path.join(
        eval_mode, 'Recall'), metrics.recall(), epoch)
    writer.add_scalar(os.path.join(
        eval_mode, 'F1Score'), metrics.f1score(), epoch)


def test_on_field_data(model, data:dict, infer_size, writer, epoch, device, test_data_label=None, tag='student model', mode='3d'):
    
    data_name, seismic, label = data['name'], data['seismic'], data['label']
    
    model.eval()
    
    if data_name == 'Equinor':
        slice_ids = cfg.equinor_slice_ids 
    elif data_name == 'Thebe':
        slice_ids = cfg.TheBe_slice_ids
    else:
        raise NotImplementedError
    
    # show field preds
    logger.info('Inferencing ' + data_name + ' ....')
    
    if mode == '3d':
        results = model_infer(model, 
                                    seismic, 
                                    infer_size, 
                                    device)
    else:
        results = model_infer_2d(model, seismic, infer_size, device) 
        
        
    prob2board(seismic, 
                    results, 
                    slice_ids, 
                    writer, 
                    os.path.join(tag + ' test result', data_name), 
                    epoch,
                    remain_prob=True)
    
    if label is not None:
        cal_metric2board(results, label, writer, epoch, tag=tag)


def sigmoid_rampup(cur_iter, rampup_iters):
    if rampup_iters == 0:
        return 1.0
    else:
        cur_iter = np.clip(cur_iter, 0.0, rampup_iters)
        phase = 1.0 - cur_iter / rampup_iters
        return float(np.exp(-5.0 * phase * phase))

def get_distance_weights(lb_slice_ids, H=128, sigma=1.0, min_weight=1e-5):
    all_indices = np.arange(H)

    lb_slice_ids = np.array(lb_slice_ids)
    dists = np.abs(all_indices[:, None] - lb_slice_ids[None, :])
    min_dists = dists.min(axis=1)
    weights = np.exp(-(min_dists**2) / (2 * sigma**2))
    
    weights = np.maximum(weights, min_weight)
    return weights.reshape(1, H, 1)

def generate_attn_map(label, sigma=10, epsilon=1e-10):
    label = (label > 0).astype(np.float32)
    
    dist_map = distance_transform_edt(1 - label)
    
    gaussian_weight = np.exp(-(dist_map ** 2) / (2 * sigma ** 2))
    
    weight_map = gaussian_weight + epsilon
    weight_map = np.clip(weight_map, 0, 1.0)
    weight_map[label == 1] = 1.0
    return weight_map.astype(np.float32)