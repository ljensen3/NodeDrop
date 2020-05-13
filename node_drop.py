import argparse
import sys
import time
import os
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, transforms

import misc
from misc import bcolors

from DataLoaders import DataLoader

from run_wrapper import RunWrapper
from network import AutoFCNetwork
from network import AutoConvNetwork
from network import ClassifyConvNetwork
from network import VGG16
from network import ResidualConvNetwork
from network import DenseNet
from network import ResNet152
def main(args):

    parser = argparse.ArgumentParser()

    parser.add_argument('run_name', metavar='N', type=str, help='name of run')
    parser.add_argument('network_type', metavar='N', type=str, help='name of run')
    parser.add_argument('gpu_id', metavar='G', type=str, help='which gpu to use')
    parser.add_argument('--print_every', metavar='N', type=int, help='number of iterations before printing', default=-1)
    parser.add_argument('--print_network', action='store_true', help='print_network for debugging')
    parser.add_argument('--data_parallel', type=int, nargs='+', default=None, help='paralellize across multiple gpus')

    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--test_print', action='store_true', help='test')
    parser.add_argument('--valid_iters', metavar='I', type=int, default=100, help='number of validation iters to run every epoch')
    parser.add_argument('--csv_file', metavar='CSV', type=str, default=None, help='name of csv file to write to')

    parser.add_argument('--sweep_lambda', action='store_true', help='preform a sweep over lambda values keeping other settings as in args')
    parser.add_argument('--sweep_c', action='store_true', help='preform a sweep over lambda values keeping other settings as in args')
    parser.add_argument('--sweep_start', metavar='S', type=float, default=0.0, help='lambda value to start sweep at')
    parser.add_argument('--sweep_stop', metavar='E', type=float, default=0.1, help='lambda value to stop sweep at')
    parser.add_argument('--sweep_step', metavar='E', type=float, default=0.01, help='step_size_between_sweep_points')
    parser.add_argument('--sweep_exp', action='store_true', help='step_size_between_sweep_points')
    parser.add_argument('--sweep_resume', action='store_true', help='resume sweep checkpts')
    parser.add_argument('--sweep_con_runs', metavar='C', type=int, default=1, help='number of runs to run with same parameters to validate constiancy')

    #checkpoints
    parser.add_argument('--checkpoint_every', type=int, default=10, help='checkpoint every n epochs')
    parser.add_argument('--load_checkpoint', action='store_true', help='load checkpoint with same name')
    parser.add_argument('--resume', action='store_true', help='resume from epoch we left off of when loading')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint to load')

    #params
    parser.add_argument('--epochs', metavar='N', type=int, help='number of epochs to run for', default=50)
    parser.add_argument('--batch_size', metavar='bs', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', metavar='lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--rmsprop', action='store_true', help='use rmsprop optimizer')
    parser.add_argument('--sgd', action='store_true', help='use sgd optimizer')
    parser.add_argument('--lr_reduce_on_plateau', action='store_true', help='update optimizer on plateau')
    parser.add_argument('--lr_exp', action='store_true', help='update optimizer on plateau')
    parser.add_argument('--lr_step', type=int, nargs='+', default=None, help='decrease lr by gamma = 0.1 on these epochs')
    parser.add_argument('--lr_list', type=float, nargs='+', default=None, help='decrease lr by gamma = 0.1 on these epochs')

    parser.add_argument('--l2_reg', type=float, default=0.0)

    DataLoader.add_args(parser)

    #added so default argument come from the network that is loaded
    network_type = args[1]
    if network_type == 'auto_fc':
        AutoFCNetwork.add_args(parser)
        network_class = AutoFCNetwork
    elif network_type == 'auto_conv':
        AutoConvNetwork.add_args(parser)
        network_class = AutoConvNetwork
    elif network_type == 'class_conv':
        ClassifyConvNetwork.add_args(parser)
        network_class = ClassifyConvNetwork
    elif network_type == 'vgg':
        VGG16.add_args(parser)
        network_class = VGG16
    elif network_type == 'res':
        ResidualConvNetwork.add_args(parser)
        network_class = ResidualConvNetwork
    elif network_type == 'res152':
        ResNet152.add_args(parser)
        network_class = ResNet152
    elif network_type == 'dense':
        DenseNet.add_args(parser)
        network_class = DenseNet
    else:
        raise ValueError('unknown network type'+str(network_type))

    args = parser.parse_args(args)

    #***************
    # GPU
    #***************

    if args.data_parallel is not None:
        try:
            del os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            pass
        device = torch.device('cuda:%d'%args.data_parallel[0])
    elif args.gpu_id == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device('cpu')
    else:
        print(bcolors.OKBLUE + 'Using GPU' + str(args.gpu_id) + bcolors.ENDC)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        device = torch.device('cuda')

    #***************
    # Run
    #***************

    if args.sweep_lambda or args.sweep_c:
        run_dir = args.run_name
        run_ckpt_dir = 'ckpts/%s'%run_dir
        if not os.path.isdir(run_ckpt_dir):
            os.mkdir(run_ckpt_dir)
        val_l=[]
        loss_l = []
        op_loss_l = []
        reg_loss_l = []
        nodes_l = []
        images_l = []

        val = args.sweep_start
        while val <= args.sweep_stop:
            print(bcolors.OKBLUE+'Val: %0.1E'%val+bcolors.ENDC)
            args.run_name = run_name = "%s/l%0.1E"%(run_dir, val)

            for i in range(args.sweep_con_runs):
                if args.sweep_con_runs > 0:
                    args.run_name = run_name+'_'+str(i)

                if not (args.sweep_resume and os.path.isdir('ckpts/'+args.run_name)):

                    print(bcolors.OKBLUE+'Run: %s'%args.run_name+bcolors.ENDC)

                    if args.sweep_lambda:
                        args.reg_lambda = float(val)
                    elif args.sweep_c:
                        args.reg_c = float(val)
                    run_wrapper = RunWrapper(args, network_class, device)

                    if not args.test:
                        run_wrapper.train()

                    loss, op_loss, reg_loss, rem_nodes,acc = run_wrapper.test(load=args.test)
                    if isinstance(rem_nodes,list):
                        rem_nodes=sum(rem_nodes)
                    x, y_hat = run_wrapper.test_print(plot=False, load=False)

                    val_l.append(val)
                    loss_l.append(loss)
                    op_loss_l.append(op_loss)
                    reg_loss_l.append(reg_loss)
                    nodes_l.append(rem_nodes)
                    if images_l == []:
                        images_l.append(x[:17])
                    images_l.append(y_hat[:17])

                    #hopefully this cleans up the gpu memory
                    del run_wrapper

            if args.sweep_exp:
                if val == 0.0:
                    val = args.sweep_start
                else:
                    val = val * args.sweep_step
            else:
                val = val + args.sweep_step 

        loss_l = np.array(loss_l)
        op_loss_l = np.array(op_loss_l)
        reg_loss_l = np.array(reg_loss_l)
        nodes_l = np.array(nodes_l)
        if args.sweep_lambda:
            lc_l = val_l#[l * args.reg_C for l in val_l]
        elif args.sweep_c:
            lc_l = [c * args.reg_lambda for c in val_l]

        print(lc_l)
        print(nodes_l)

        misc.plot_sweep(run_ckpt_dir, lc_l, op_loss_l, nodes_l)
        #misc.sweep_to_image(images_l, run_ckpt_dir)

    else:
        #default single run behavior
        run_wrapper = RunWrapper(args, network_class, device)

        if args.test_print:
            run_wrapper.test_print()
        elif args.test:
            run_wrapper.test()
        else:
            run_wrapper.train()
    


if __name__ == "__main__":
    main(sys.argv[1:])
