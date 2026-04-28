import torch 
from model.vnet import VNet
from data.dataset import Synthetic
from torch.utils.data import DataLoader
import cfg
from utils.metric import MMetric
from tqdm import tqdm
import numpy as np 
import cigvis 
from cigvis import colormap
from matplotlib import pyplot as plt


def evaluate(model, val_loader, device):
    model.eval()
    metrics = MMetric()

    val_loader = tqdm(val_loader)
    
    with torch.amp.autocast(device_type='cuda'):
        Seis = []
        Ps = []
        GTs = []
        for idx, data in enumerate(val_loader):
            seismic = data['seis'].to(device)
            fault = data['fault']
            
            with torch.no_grad():
                pred = model(seismic, turnoff_drop=True)['logits']
                
            pred = torch.argmax(torch.softmax(pred.detach(), dim=1), dim=1, keepdim=True)  # [b,1,d,h,w]
            pred = pred.cpu().numpy()[:, 0]

            for j in range(pred.shape[0]):
                Seis.append(seismic.cpu().numpy()[j, 0])
                Ps.append(pred[j])
                GTs.append(fault.numpy()[j])

        Seis = np.concatenate(Seis, axis=2)
        Ps = np.concatenate(Ps, axis=2)
        GTs = np.concatenate(GTs, axis=2)
        metrics.update(Ps, GTs)
        
    # metrics to tensorboard
    eval_metrics = ' DSC {:.4f}, F1 {:.4f}, IoU {:.4f}, Precision {:.4f}, Recall {:.4f}'.format(metrics.dice(), metrics.f1score(), metrics.iou(), metrics.precision(), metrics.recall())
    return eval_metrics, Seis, Ps, GTs


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ckpts = {
        'lb_32': 'ckpts/TTCT_lb_32.pt',
        'lb_16': 'ckpts/TTCT_lb_16.pt',
        'lb_8': 'ckpts/TTCT_lb_8.pt',
        'lb_8_wo_aba': 'ckpts/TTCT_wo_aba_loss.pt',
        'lb_8_aba': 'ckpts/TTCT_aba_loss_sigma2_7.pt',
    }
    results = {}
    
    for model_name, ckpts in ckpts.items():
        
        model = VNet().to(device)
        state_dict = torch.load(ckpts, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model3d'])
        valid_set = Synthetic(root=cfg.synthetic_root_dir, train=False)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
        
        eval_metrics, seiss, preds, gts = evaluate(model, valid_loader, device)
        
        print('Model {} results: {}'.format(model_name, eval_metrics))
        results.update({model_name: {'seis': seiss, 'preds': preds, 'gts': gts}})
    
    sample_id = 9
    slice_id = 48
    
    vis_node_lst = []
    fg_cmap = colormap.set_alpha_except_min('jet', 1.)
    for idx, (model_name, res) in enumerate(results.items()):
        seis = res['seis']
        preds = res['preds']
        gts = res['gts']
        seis = seiss[:, :, sample_id*128:(sample_id+1)*128]
        pred = preds[:, :, sample_id*128:(sample_id+1)*128]
        gt = gts[:, :, sample_id*128:(sample_id+1)*128]
        # 3d visualization of prediction
        seis_node = cigvis.create_slices(seis, cmap='gray')
        vis_node_lst.append(cigvis.add_mask(seis_node, pred.astype(np.float32), cmaps=fg_cmap, interpolation='nearest'))
        # 2d visualization of prediction
        plt.subplot(1, len(results) + 1, idx+1)
        plt.title(model_name,)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(pred[slice_id], cmap='jet')
    
    # 3d visualization of gt
    seis_node = cigvis.create_slices(seis, cmap='gray')
    vis_node_lst.append(cigvis.add_mask(seis_node, gt.astype(np.float32), cmaps=fg_cmap, interpolation='nearest'))
    cigvis.plot3D(vis_node_lst, grid=(1, len(results) + 1), share=True)
    # 2d visualization of gt
    plt.subplot(1, len(results) + 1, len(results)+1)
    plt.title('gt')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gt[slice_id], cmap='jet')
    plt.tight_layout()
    plt.savefig('result.png', dpi=300)
    
    