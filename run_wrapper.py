import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import numpy as np
import os
import csv
import time
import functools
from summary_prints import SummaryPrint
from misc import bcolors
import misc

from DataLoaders import DataLoader

class RunWrapper(object):
    def __init__(self, args, network_class, device):
        super(RunWrapper, self).__init__()

        self.args = args
        self.device = device

        #Misc init
        self.epoch = 0

        self.ckpt_dir = 'ckpts/{0}'.format(args.run_name)
        if not os.path.isdir('ckpts'):
            os.mkdir('ckpts')

        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        #network
        print('Creating Dataloader')
        data_loader = DataLoader(args, test=False)
        data = next(iter(data_loader))
        x,y = data
        x_size = x.shape

        print('Creating Network')
        self.network = network_class(args, x_size)
        if args.print_network:
            print(self.network)

        #summary handler
        self.sumPrint = SummaryPrint(args,self.network.loss_names(), self.ckpt_dir, 'train' if not args.test else 'test')
        if not args.test:
            self.validSumPrint = SummaryPrint(args,self.network.loss_names(), self.ckpt_dir, 'valid', color=bcolors.OKBLUE)

        #send to GPU
        print(bcolors.OKBLUE + 'Moving to specified device' + bcolors.ENDC)
        #self.network = torch.nn.DataParallel(self.network).cuda()
        if self.args.data_parallel is not None:
            self.network = torch.nn.DataParallel(self.network, device_ids=self.args.data_parallel, output_device=self.args.data_parallel[0])
            self.network.cuda(self.args.data_parallel[0])
        else:
            self.network = self.network.to(device)
        cudnn.benchmark = True

        #optimizer
        if self.args.rmsprop:
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=args.lr, weight_decay=self.args.l2_reg)
        elif self.args.sgd:
            self.optimizer = optim.SGD(self.network.parameters(), lr=args.lr, momentum=0.9, weight_decay=self.args.l2_reg, nesterov=True)
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=args.lr, weight_decay=self.args.l2_reg)
            
        #lr scheduler
        if args.lr_reduce_on_plateau:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, verbose=True, threshold=0.01, cooldown=50, min_lr=1e-6)
        elif args.lr_exp:
            def lr_scheduler(epoch):
                lr_fact = (0.8 ** (float(epoch) / 20))
                return lr_fact
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_scheduler)
        elif args.lr_list:
            self.lr = 1.0
            self.lr_map = {int(self.args.lr_list[i]):self.args.lr_list[i+1] for i in range(0, len(self.args.lr_list), 2)}
            def lr_scheduler(epoch):
                if epoch in self.lr_map:
                    self.lr = self.lr_map[epoch]
                return self.lr
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_scheduler)
        elif args.lr_step is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, args.lr_step, gamma=0.1)
        else:
            self.lr_scheduler = None

    def load(self, resume=False):
        #load checkpoints
        if self.args.checkpoint is None:
            print(bcolors.OKBLUE + 'Loading Checkpoint: ' + self.args.run_name + bcolors.ENDC)
            checkpoint = torch.load(self.ckpt_dir+"/ckpt.pth")
        else:
            print(bcolors.OKBLUE + 'Loading Checkpoint: ' + self.args.checkpoint + bcolors.ENDC)
            checkpoint = torch.load("ckpts/%s/ckpt.pth" % self.args.checkpoint)
        if self.args.data_parallel is not None:
            self.network.module.load_state_dict(checkpoint['network'])
        else:
            self.network.load_state_dict(checkpoint['network'])
        if resume:
            self.optimizer.load_state_dict(checkpoint['opt'])
            self.epoch = checkpoint['epoch']
        print(bcolors.OKBLUE + 'Finished Loading Checkpoint ' + bcolors.ENDC)

    def save(self):
        print(bcolors.OKBLUE + 'Saving Checkpoint: ' + self.args.run_name + bcolors.ENDC)
        torch.save({
            'network':(self.network.state_dict() if self.args.data_parallel is None else self.network.module.state_dict()),
            'opt':self.optimizer.state_dict(),
            'epoch':self.epoch+1,
            'args':self.args,
            }, self.ckpt_dir+"/ckpt.pth")

    def _iter(self, x, y, sumPrint, backwards=True):
        if not self.args.data_parallel:
            x, y = x.to(self.device), y.to(self.device)              

        y_bar = self.network(x)
        if not self.args.data_parallel:
            loss_l = self.network.loss(x,y,y_bar)
        else:
            z,y = x.to(y_bar.device), y.to(y_bar.device)
            loss_l = self.network.module.loss(x,y,y_bar)

        if backwards:
            self.optimizer.zero_grad()
            loss_l[0].backward()
            self.optimizer.step()
        try: 
            return [l.data.item() for l in loss_l]
        except:
            ls1=[]
            for l in loss_l:
                try:
                    ls1+=[l.data.item()]
                except:
                    ls1+=[l]
            #print(ls1)
            return ls1

    def run_epoch(self, data_loader, test=False):
        self.sumPrint.start_epoch(self.epoch, len(data_loader))
        for j, (data, target) in enumerate(data_loader):
            if data is None or target is None:
                continue
            self.sumPrint.start_iter(j)
            res = self._iter(data, target, self.sumPrint, backwards=not test)
            self.sumPrint.end_iter(j, res)
        print('A',self.network)
        if(self.args.data_parallel is not None):
            act_nodes = self.network.module.active_nodes()
        else:
            act_nodes = self.network.active_nodes()
        rets = self.sumPrint.end_epoch({'act nodes':act_nodes})

        if not test:
            self.network.eval()
            self.validSumPrint.start_epoch(self.epoch, len(data_loader.valid_data_loader))
            for j, (data, target) in enumerate(data_loader.valid_data_loader):
                self.validSumPrint.start_iter(j)
                res = self._iter(data, target, self.validSumPrint, backwards=False)
                self.validSumPrint.end_iter(j, res)
            self.network.train()

            val_rets = self.validSumPrint.end_epoch()
        else:
            val_rets = None

        return rets, val_rets

    def test_print(self, plot=True, load=True):
        print(bcolors.OKBLUE+'*******PRINTING********'+bcolors.ENDC)
        data_loader = DataLoader(self.args, test=True)
        if load:
            self.load()
        self.network.node_drop()
        x,y = next(iter(data_loader))
        x = x.to(self.device)
        y_hat = self.network(x)
        if plot:
            misc.batch_to_image(x,y_hat)

        return x.detach().cpu().numpy(), y_hat.detach().cpu().numpy()

    def _count_parameters(self):
        init_params = 0
        for mod in self.network.modules():
            if hasattr(mod, 'weight') and mod.weight is not None:
                if mod.weight.dim() != 0:
                    init_params += functools.reduce(lambda x, y: x*y, mod.weight.size())
            if hasattr(mod, 'bias') and mod.bias is not None:
                if mod.bias.dim() != 0:
                    init_params += mod.bias.size(0)
        return init_params

    def test(self, load=True):
        #testing
        print(bcolors.OKBLUE+'*******TESTING********'+bcolors.ENDC)
        data_loader = DataLoader(self.args, test=True)
        #load checkpoint
        if load:
            self.load()
        #total parameters
        init_params = self._count_parameters()
        #drop nodes
        rem_nodes = self.network.node_drop()
        rem_params = self._count_parameters()
        print(rem_params)
        #set no gradients
        self.network.eval()
        #run epoch
        rets, _ = self.run_epoch(data_loader, True)
        rets = [self.args.run_name] + rets #run name
        #sum of nodes
        rmt = torch.sum(torch.stack(rem_nodes)).detach().cpu().numpy()
        #number of parameters
        rem_params = self._count_parameters()
        red_per = float(rem_params)/float(init_params)
        red_factor = float(init_params)/float(rem_params)
        rets += [init_params, rem_params, red_per, red_factor, int(rmt), str([int(rm.detach().cpu().numpy()) for rm in rem_nodes])]
        print(rets)
        #write csv
        if self.args.csv_file is not None:
            csv_file = 'csv/%s.csv'%self.args.csv_file
            if not os.path.isfile(csv_file):
                #write header
                with open(csv_file, 'w') as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(['name'] + self.network.loss_names() + ['init params', 'final params', 'reduction per', 'ref factor', 'rem nodes', 'rem nodes layerwise'])
            with open(csv_file, 'a') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(rets)

        return rets

    def train(self):
        if self.args.load_checkpoint or self.args.resume: # or (self.args.checkpoint is not None):
            self.load(self.args.resume)
        data_loader = DataLoader(self.args, test=False)

        print(bcolors.OKBLUE+'*******TRAINING********'+bcolors.ENDC)
        #set requires gradients
        self.network.train()
        #run epochs
        while self.epoch < self.args.epochs:
            _, val_ret = self.run_epoch(data_loader, False)
            if(self.args.data_parallel is None):
                self.network.epoch_callback(self.epoch)
            else:
                self.network.module.epoch_callback(self.epoch)
            self.epoch += 1
            if self.epoch % self.args.checkpoint_every == 0:
                self.save()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        #save
        self.save()
