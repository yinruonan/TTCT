import argparse
import cfg
import logging

logger = logging.getLogger(__name__)

class TrainArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # data related
        self.parser.add_argument('--batchsz', type=int, default=4, help='batch size')
        self.parser.add_argument('--num-workers', type=int, default=0)

        # optimizer related
        self.parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        self.parser.add_argument('--power', type=float, default=0.9, help='lr scheduler parameters')
        self.parser.add_argument('--weight-decay', type=float, default=5e-4)
        # train related
        self.parser.add_argument('--root-dir', type=str, default=cfg.synthetic_root_dir)
        self.parser.add_argument('--log-dir', type=str, default='log')
        self.parser.add_argument('--test-data', type=str, default='')
        self.parser.add_argument('--alg', type=str, default='ttct')
        self.parser.add_argument('--resume', type=str, default=None)
        self.parser.add_argument('--sim-expt-ann', action="store_true")
        self.parser.add_argument('--sigma_1', type=float, default=2.0)
        self.parser.add_argument('--sigma_2', type=float, default=7.0)

        self.parser.add_argument('--exp-name', type=str, default='')
        self.parser.add_argument('--n-slice', type=int, default=8)
        self.parser.add_argument('--epochs', type=int, default=1000, help='training epochs')
        self.parser.add_argument('--ckpts-dir', type=str, default='ckpts', help='directory to save checkpoints')
        self.parser.add_argument('--eval-freq', type=int, default=20, help='evaluate frequency')
        self.parser.add_argument('--save-freq', type=int, default=20, help='evaluate frequency')


    def parse(self):
        opt = self.parser.parse_args()
        logger.info('---- load TTCT options ----')
        for name, val in sorted(vars(opt).items()):
            logger.info('\t{}: {}'.format(str(name), str(val)))
        logger.info(' ---- ======================= ----\n')
        
        return opt