import numpy as np
import random
import torch
import os
import shutil
import time

from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import RMSprop
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from agents.base import BaseAgent
from graphs.models.inception_v3 import inception_v3
from datasets.inception_v3 import inception_data
from utils.misc import get_device
from utils.utils import read_image

class inception_v3_agent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = get_device()

        # define models
        # inception_v3 input size = ( N x 3 x 299 x 299 )
        self.model = inception_v3(pretrained=True, num_classes=cfg.num_classes)

        # define data_loader
        self.dataset = inception_data(cfg).get_dataset()
        tr_size = int(cfg.train_test_ratio * len(self.dataset))
        te_size = len(self.dataset) - tr_size
        tr_dataset, te_dataset = random_split(self.dataset, [tr_size, te_size])
        self.tr_loader = DataLoader(tr_dataset, batch_size=cfg.bs, 
                        shuffle=cfg.data_shuffle, num_workers=cfg.num_workers)
        self.te_loader = DataLoader(te_dataset, batch_size=cfg.bs, 
                        shuffle=cfg.data_shuffle, num_workers=cfg.num_workers)

        # define loss
        self.loss = torch.tensor(0)
        self.criterion = CrossEntropyLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = RMSprop(self.model.parameters(), lr=cfg.lr)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        self.best_info = ""

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.cfg.cuda:
            self.logger.info(
        "WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.cfg.cuda

        # set the manual seed for torch
        self.manual_seed = self.cfg.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.model = self.model.to(self.device)
            if self.cfg.data_parallel:
                self.model = nn.DataParallel(self.model)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
        else:
            self.model = self.model.to(self.device)
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from cfg if not found start from scratch.
        self.exp_dir = os.path.join('./experiments', cfg.exp_name)
        self.load_checkpoint(self.cfg.checkpoint_filename)
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

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            info = "Checkpoint loaded successfully from "
            self.logger.info(info + "'{}' at (epoch {}) at (iteration {})\n"
              .format(file_name, checkpoint['epoch'], checkpoint['iteration']))
                
        except OSError as e:
            self.logger.info("Checkpoint not found in '{}'".format(file_name))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
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

        if is_best:
            shutil.copyfile(os.path.join(checkpoint_dir, file_name),
                            os.path.join(checkpoint_dir, 'best.pt'))

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.cfg.mode == 'train':
                self.train()
            elif self.cfg.mode == 'predict':
                self.predict()
            else:
                self.logger.info("\'mode\' value of cfg file is wrong")
                raise ValueError
            
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")


    def train(self):
        """
        Main training loop
        :return:
        """
        self.validate()
        for e in range(1, self.cfg.epochs+1):
            self.current_epoch = e 
            self.train_one_epoch()
            self.validate()
        print(self.best_info)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        for batch_idx, (imgs, labels) in enumerate(self.tr_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs, aux_outputs = self.model(imgs).values()
            loss1 = self.criterion(outputs, labels)
            loss2 = self.criterion(aux_outputs, labels)
            self.loss = loss1 + 0.3*loss2

            _, preds = torch.max(outputs, 1)
            acc = preds.eq(labels.view_as(preds)).sum().item() / self.cfg.bs

            self.loss.backward()
            self.optimizer.step()
            
            self.summary_writer.add_scalars('scalar_group', 
                        {   'loss_end' : loss1.item(),
                            'loss_aux' : loss2.item(),
                            'loss_total' : self.loss.item(),
                            'accuracy' : acc},
                            self.current_iteration)

            if batch_idx % self.cfg.log_interval == 0:
                info_1 = 'Epochs {} [{}/{} ({:.0f}%)] | Loss: {:.6f}' .format(
                            self.current_epoch, 
                            batch_idx * len(imgs), 
                            len(self.tr_loader.dataset), 
                            100. * batch_idx / len(self.tr_loader),
                            self.loss.item())
                info_2 = 'Batch Accuracy : {:.2f}'.format(acc)
                self.logger.info('{} | {}'.format(info_1, info_2))
                self.save_checkpoint('{}_epoch{}_iter{}.pt'.format(
                                                self.cfg.exp_name,
                                                self.current_epoch, 
                                                self.current_iteration)
                                                )
            self.current_iteration += 1

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(self.te_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs, aux_outputs = self.model(imgs).values()
                loss1 = self.criterion(outputs, labels)
                loss2 = self.criterion(aux_outputs, labels)
                test_loss += loss1 + 0.3*loss2
                
                # get the index of the max log-probability
                _, preds = torch.max(outputs, 1)
                correct += preds.eq(labels.view_as(preds)).sum().item()

        test_loss /= len(self.te_loader)
        acc = correct / (len(self.te_loader)*self.cfg.bs)
        self.logger.info('Test: Avg loss:{:.4f}, Accuracy:{}/{} ({:.2f}%)\n'
                .format(test_loss,
                        correct, 
                        len(self.te_loader)*self.cfg.bs,
                        100*acc))
        if self.best_metric <= acc:
            self.best_metric = acc
            self.best_info = 'Best: {}_epoch{}_iter{}.pt'.format(
                                        self.cfg.exp_name,
                                        self.current_epoch, 
                                        self.current_iteration-1)
    
    def predict(self):
        try:
            from tkinter.filedialog import askdirectory
            from glob import glob
            directory = askdirectory(title="select a directory")
            fn_list = glob(directory+'/*.jpg')
        except ImportError:
            from glob import glob
            fn_list = glob(self.cfg.test_img_path+'/*.jpg')

        result = {k: 0 for k in self.dataset.classes}
        t = time.time()
        for fn in fn_list:
            img = read_image(fn, size=(299,299))
            img = img.to(self.device)
            self.model.eval()
            output = self.model(img)

            _, pred = torch.max(output, 1)
            res = self.dataset.idx_to_class[pred.item()]
            print("{} : {}".format(fn, res))
            result[res] += 1
        print("result : ", result)
        print('process spend : {} sec for {} images'.format(time.time()-t, len(fn_list)))

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, 
        the operator and the data loader
        :return:
        """
        pass
