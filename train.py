import utils.func as func
import numpy as np 
from data.dataset import get_dataset
from torch.utils.data import DataLoader
import os 

import torch 
from utils import criterions
from args import TrainArgs
import logging
from exp.baseline import Baseline
from exp.ttct import TTCT

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- setup logging ---
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    # manual seed
    func.seed_all(10086)
    # train options
    args = TrainArgs().parse()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # test on field data during training.
    # test_data_name = args.test_data.split(os.path.sep)[-1].split('.')[0]
    # test_data = np.load(args.test_data)
    # test_data = {'name': test_data_name, 
    #                    'seismic': test_data}

    criterion = criterions.CELoss()
    # tensorboard writer
    writer = func.get_writer(args)
    cross_annot = False
    if args.alg == 'baseline':
        alg = Baseline(args, criterion=criterion, device=device)
    
    elif args.alg == 'ttct':
        criterion = criterions.CompoundLoss()   
        alg = TTCT(args, criterion, device, writer) 
    else:
        raise NotImplementedError('algorithm {} is not implemented.'.format(args.alg))
    
    if args.resume is not None:
        logger.info('======== load pretrained model {} ========'.format(args.resume))
        alg.load_ckpts(args.resume)

    # prepare dataloaders.
    sim_expt_ann = args.sim_expt_ann
    datasets = get_dataset(root=args.root_dir, n_slice=args.n_slice, cross_annot=cross_annot, sim_expt_ann=sim_expt_ann, sigma_1=args.sigma_1, sigma_2=args.sigma_2)
    trainset = datasets['trainset']
    validset = datasets['validset']
    
    trainloader = DataLoader(trainset, args.batchsz, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    validloader = DataLoader(validset, args.batchsz, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # setup lr scheduler
    alg.max_it = args.epochs * len(trainloader)
    alg.set_scheduler()
    
    args.rampup = int(alg.max_it * 0.5)

    # train
    for epoch in range(0, args.epochs):
        for idx, data in enumerate(trainloader):
            
            alg.train_iter(data)
            alg.update_lr()
            # show losses
            it_loss = alg.log2board(writer, tag='it')
            
            logger.info('Epoch {:3d}/{:3d}, global step {:6d} : '.format(epoch + 1, args.epochs, alg.it) + it_loss)

        # show epoch average loss & save checkpoints.
        epoch_loss = alg.log2board(writer, tag='avg', epoch=epoch+1)
        
        logger.info('Epoch {} complete: average '.format(epoch + 1) + epoch_loss)
        if (epoch + 1) % args.save_freq == 0:
            logger.info(' ---- save the model @ epoch {} ----'.format(epoch + 1)) 
            alg.save_ckpt('Epoch_{}.pt'.format(epoch + 1))

        if (epoch + 1) % args.eval_freq == 0:
            
            func.evaluate(model=alg.get_seg_model(), val_loader=validloader, writer=writer, epoch=epoch+1, device=device)
            
            if args.alg == 'TTCT':
                func.evaluate(model=alg.get_seg_model2d(), val_loader=validloader, writer=writer, epoch=epoch+1, device=device, is_3d=False)
            
            # func.test_on_field_data(alg.get_seg_model(), data=test_data, infer_size=128, writer=writer, epoch=epoch+1, device=device)
            # func.test_on_field_data(alg.get_t_seg_model(), data=test_data, infer_size=128, writer=writer, epoch=epoch+1, device=device, tag='teacher model')

    logger.info('---- train complete. ----')



if __name__ == '__main__':
    main()
