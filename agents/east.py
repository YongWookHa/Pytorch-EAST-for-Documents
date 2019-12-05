import numpy as np
import shutil
import os
import time
import torch
from tqdm import tqdm
from PIL import Image

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

from tensorboardX import SummaryWriter

from agents.base import BaseAgent
from agents.eval import east_eval
from datasets.east import custom_dataset
from graphs.models.east import EAST
from graphs.losses import EASTLoss
from utils.misc import print_cuda_statistics, get_device, timeit

cudnn.benchmark = True


class EAST_agent(BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = get_device()
        
        # define models
        self.model = EAST(pretrained=True, cfg=cfg)

        # set cuda flag
        self.cuda = (self.device == torch.device('cuda')) and cfg.cuda

        # set the manual seed for torch
        self.manual_seed = cfg.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            self.model = self.model.to(self.device)
            if cfg.data_parallel:
                self.model = nn.DataParallel(self.model)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
        else:
            self.model = self.model.to(torch.device('cpu'))
            self.logger.info("Program will run on *****CPU*****\n")

        # ready done for prediction
        if cfg.mode == "predict":
            self.load_checkpoint(cfg.checkpoint_file)
            self.model.eval()
            return

        # define data_loader
        train_dataset = custom_dataset(cfg.tr_im_pth, cfg.tr_gt_pth)
        test_dataset = custom_dataset(cfg.te_im_pth, cfg.te_gt_pth)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.bs, 
              shuffle=cfg.data_shuffle, num_workers=cfg.num_w, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=cfg.bs, 
              shuffle=cfg.data_shuffle, num_workers=cfg.num_w, drop_last=True)

        # define loss
        self.criterion = EASTLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, 
                                  milestones=[cfg.epochs//2], gamma=0.1)

        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 1
        self.best_metric = np.inf  # loss
        self.best_info = ''

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.exp_dir = os.path.join('./experiments', cfg.exp_name)
        self.load_checkpoint(cfg.checkpoint_file)
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.exp_dir,
                                                              'summaries'))

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            self.logger.info("Loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name, map_location=self.device)

            self.model.load_state_dict(checkpoint['model'], strict=False)

            if self.cfg.mode != "predict":
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            info = "Checkpoint loaded successfully from "
            self.logger.info(info + "'{}' at (epoch {}) at (iteration {})\n"
              .format(file_name, checkpoint['epoch'], checkpoint['iteration']))
                
        except OSError as e:
            self.logger.info("Checkpoint not found in '{}'.".format(file_name))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current 
                        checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'model' : self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        # save the state
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        torch.save(state, os.path.join(checkpoint_dir, file_name))

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.cfg.mode == 'train':
                self.model.train()
                self.train()
            elif self.cfg.mode == 'predict':
                self.predict()
            else:
                raise ValueError("Invalid mode")

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for e in range(self.current_epoch, self.cfg.epochs+1):
            self.logger.info('----- Train Epochs:[{}/{}] -----'.format(
                                        self.current_epoch, self.cfg.epochs))
            self.train_one_epoch()
            if self.current_epoch % self.cfg.save_per_epoch == 0:
                self.logger.info('----- Test -----')
                self.validate()
            self.current_epoch += 1

    @timeit
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        epoch_loss = 0
        tqdm_bar = tqdm(  enumerate(self.train_loader, 1), 
                          total=len(self.train_loader),
                          desc="TRAIN Loss : XX.XX"
                       )
        
        for i, d in tqdm_bar:
            self.optimizer.zero_grad()

            img, gt_score = d[0].to(self.device), d[1].to(self.device)
            gt_geo, ignored_map = d[2].to(self.device), d[3].to(self.device)

            pred_score, pred_geo = self.model(img)
            losses = self.criterion(gt_score, pred_score, 
                                  gt_geo, pred_geo, ignored_map)
            loss = losses['geo_loss'] + losses['classify_loss']
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            self.summary_writer.add_scalars(
                        'scalar_group', 
                        {
                            'classify_loss' : losses['classify_loss'].item(),
                            'iou_loss' : losses['iou_loss'].item(),
                            'angle_loss' : losses['angle_loss'].item(),
                            'geo_loss' : losses['geo_loss'].item(),
                            'train_loss' : loss.item(),
                        }, 
                        self.current_iteration)
            tqdm_bar.set_description("TRAIN Loss : {:.2f}".format(loss.item()))
            self.current_iteration += 1

        self.logger.info('Epochs:[{}/{}], avg_loss:{:.8f}'.format(
                            self.current_epoch, self.cfg.epochs, epoch_loss/i))
            
        if self.best_metric > epoch_loss / i:
                self.best_metric = epoch_loss / i
                # self.save_checkpoint('best.pt')
                
                self.best_info = 'Best: {}_epoch{}_iter{}.pt'.format(
                                self.cfg.exp_name,
                                self.current_epoch, 
                                self.current_iteration-1)
            
    @timeit
    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        with torch.no_grad():
            epoch_loss = 0
            tqdm_bar = tqdm(enumerate(self.test_loader, 1), 
                            total=len(self.test_loader),
                            desc="VALIDATE Loss : XX.XX"
                            )
            for i, d in tqdm_bar:
                img, gt_score = d[0].to(self.device), d[1].to(self.device)
                gt_geo, ignored_map = d[2].to(self.device), d[3].to(self.device)

                pred_score, pred_geo = self.model(img)
                losses = self.criterion(gt_score, pred_score, 
                                    gt_geo, pred_geo, ignored_map)
                loss = losses['geo_loss'] + losses['classify_loss']
                epoch_loss += loss.item()
                
                tqdm_bar.set_description("VALIDATE Loss : {:.2f}"
                                              .format(loss.item()))
            self.logger.info('TEST: Epochs:[{}/{}], avg_loss:{:.8f}'.format(
                                                        self.current_epoch, 
                                                        self.cfg.epochs, 
                                                        epoch_loss/i))

        if self.current_epoch % self.cfg.save_per_epoch == 0:
            self.save_checkpoint('{}_epoch{}_iter{}.pt'.format( 
                                                    self.cfg.exp_name,
                                                    self.current_epoch, 
                                                    self.current_iteration-1))
        
    def predict(self):
        from glob import glob
        img_list = glob(self.cfg.pred_img_pth+'/*.jpg')

        for img in img_list:
            img = Image.open(img)
            t= time.time()
            boxes = east_eval.detect(img, self.model, self.device)
            print('time consumption :', time.time()-t)
            print('num of boxes : ', len(boxes))
            plot_img = east_eval.plot_boxes(img, boxes)
            plot_img.save('res.jpg')

    def finalize(self):
        try:
            self.logger.info(self.best_info)
        except AttributeError:
            pass