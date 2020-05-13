import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
from misc import ClampedRelu
from misc import TrueClampedRelu
from misc import StraightThroughRelu
from misc import StraightThroughFC
from misc import bcolors
from misc import AnnealedDropout
import misc

class NetworkBase(nn.Module):
    """
    Base class for all models
    Each class should implement the functions
      Forward - standard py torch forward for network
      loss - computes the loss for the network given desired values
      node_drop - drops nodes in the network
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument('--reg_lambda',metavar='l',type=float, help='lambda for num_child loss',default=1e-2)
        parser.add_argument('--reg_C',metavar='C',type=float, help='lambda for value loss',default=1e-1)
        parser.add_argument('--reg_pow',metavar='C',type=float, help='lambda for value loss',default=1.0)

        parser.add_argument('--reg_lambda_anneal', metavar='la', type=int, help='anneal lambda up over this number of epochs', default=None)

        parser.add_argument('--l2_lambda', metavar='l', type=float, help='l22 reg lambda', default=None)
        parser.add_argument('--l2_lambda_bn', action='store_true', help='weight decay on bn layers')
        parser.add_argument('--l2_lambda_bias', action='store_true', help='weight decay on bn layers')
        parser.add_argument('--cor_lambda',metavar='l',type=float, help='lambda for corelation loss',default=0.0)

        parser.add_argument('--reg_init_epochs', type=int, default=0, help='epochs to run without reg loss before turning it on')

    def __init__(self, args):
        super(NetworkBase, self).__init__()
        self.args = args

        if args.reg_lambda_anneal is None:
            self.reg_lambda = args.reg_lambda
        else:
            self.reg_lambda = 0.0

        if (not args.test) and args.reg_init_epochs > 0:
            self.init = True
        else:
            self.init = False

    def loss(self, x, y, y_bar):
        r"""
        computes the loss for this network
        implemented by subclass
        
        Args:
        x - network input
        y - desired output
        y_bar - network output
    
        """
        raise ValueError('Implemented by Subclass')
        
    def loss_names(self):
        #return the names of the losses your loss function will return
        raise ValueError('Implemented by Subclass')

    def node_drop(self):
        r"""
        drops nodes which are no longer needed
        implemented by subclass
        """

        raise ValueError('Implemented by Subclass')

    def _node_drop_loss(self, layer):
        power=self.args.reg_pow
        
        """
        #this version computes the sum of the norms
        zero = torch.zeros((), device=layer.weight.device)
        weights = torch.where(layer.weight<=0, zero, layer.weight).view(layer.weights.size(0), -1)
        weight_l1_loss = torch.sum(torch.norm(weights, dim=-1, p=power))
        bias_l1_loss = torch.sum(torch.where(layer.bias + self.args.reg_C),p=power)
        """
        
        zero = torch.zeros((), device=layer.weight.device)
        if self.init:
            return zero

        #this version computes the norm^pow eg l2^2

        weight_l1_loss = torch.sum(torch.where(layer.weight<=0, zero, layer.weight).pow(power))
        bpc = layer.bias + self.args.reg_C
        bias_l1_loss = torch.sum(torch.where(bpc<=0, zero, bpc).pow(power))
        
        reg_loss = self.reg_lambda * (weight_l1_loss + bias_l1_loss)

        return reg_loss
                
    #extra code by im adding it this way to avoid the isinstnace call which is a little slow
    def _node_drop_loss_batch_norm(self, layer):
        zero = torch.zeros((), device=layer.weight.device)
        if self.init:
            return zero

        if self.args.batch_norm_func == 'var':
            weight_value = torch.abs(layer.weight) * np.sqrt(layer.weight.size(0))
        elif self.args.batch_norm_func == 'l2':
            weight_value = torch.abs(layer.weight)
        max_value = weight_value + layer.bias
        vpc = max_value + self.args.reg_C

        l1_loss = torch.sum(torch.where(vpc<=0, zero, vpc))
        
        reg_loss = self.reg_lambda * l1_loss

        return reg_loss

    def _cor_loss(self, output):
        if self.args.cor_lambda == 0.0:
            return 0.0
        #expects output to be 2D batch x data
        mean = torch.mean(output, dim=0, keepdim=True)
        std = torch.std(output, dim=0, keepdim=True)
        var = (output-mean)/(std+1e-15)
        mask = 1 - torch.eye(var.size(1), device=var.device).unsqueeze(0)
        cor_coef = torch.sum((torch.unsqueeze(var,1)*torch.unsqueeze(var,2)) * mask, dim=0) / (self.args.batch_size-1)
        loss = self.args.cor_lambda * torch.sum(torch.abs(cor_coef))
        if loss == 0.0:
            #check for when all nodes are off
            return 0.0

        return loss

    def node_drop_loss(self):
        if self.args.batch_norm:
            return sum([self._node_drop_loss_batch_norm(layer) for layer in self.node_drop_layers])
        else:
            return sum([self._node_drop_loss(layer) for layer in self.node_drop_layers])

    def reg_l2_loss(self):
        if self.args.l2_lambda == None:
            return torch.zeros((), device=self.reg_layers[0].weight.device)
        else:
            if self.args.l2_lambda_bias == True:
                return (self.args.l2_lambda / (2.0)) * (sum([torch.sum(layer.weight ** 2) for layer in self.reg_layers]) + sum([torch.sum(layer.bias ** 2) for layer in self.reg_layers]))
            else:
                return (self.args.l2_lambda / (2.0)) * sum([torch.sum(layer.weight ** 2) for layer in self.reg_layers])

    def _drop_nodes(self, layer_1, layer_2, conv=False):
        print(bcolors.OKGREEN + 'Dropping Nodes for layer %s'%str(layer_1) + bcolors.ENDC)
        e_w = []
        e_b = []
        d_w = []
        for i in range(layer_1.bias.size(0)):
            b = layer_1.bias[i]
            w = layer_1.weight[i,:]
            if (torch.sum(torch.max(w,torch.zeros((), device=w.device)))+b >= 0):
                e_w.append(w)
                e_b.append(b)
                if isinstance(layer_2, nn.Conv2d) and not conv:
                    #deconv layers have same shape as conv ones (not reversed)
                    d_w.append(layer_2.weight[i,:])
                else:
                    d_w.append(layer_2.weight[:,i])
        rem_nodes = len(e_b)

        if rem_nodes>0:
            layer_1.weight.data = torch.stack(e_w)
            layer_1.bias.data = torch.stack(e_b)
            if isinstance(layer_2, nn.Conv2d) and not conv:
                layer_2.weight.data = torch.stack(d_w)
            elif conv:
                layer_2.weight.data = torch.stack(d_w).permute(1,0,2,3)
            else:
                layer_2.weight.data = torch.stack(d_w).t()

        print(bcolors.OKGREEN + 'Kept %d Nodes'%(rem_nodes) + bcolors.ENDC)        
        return rem_nodes

    def _drop_batch_norm_layer(self, layer, mask):
        if layer.weight is not None:
            layer.weight.data = torch.stack([w for w, m in zip(layer.weight.data, mask) if m==1])
        if layer.bias is not None:
            layer.bias.data = torch.stack([b for b,m in zip(layer.bias.data, mask) if m==1])
        if layer.running_mean is not None:
            layer.running_mean.data = torch.stack([w for w, m in zip(layer.running_mean.data, mask) if m==1])
        if layer.running_var is not None:
            layer.running_var.data = torch.stack([w for w, m in zip(layer.running_var.data, mask) if m==1])

    def _drop_layer_output(self, layer, mask, bias=True):
        if torch.sum(mask) == 0:
            layer.weight.data = torch.zeros((), device=layer.weight.data.device)
            layer.bias.data = torch.zeros((), device=layer.bias.data.device)
        else:
            layer.weight.data = torch.stack([w for w, m in zip(layer.weight.data, mask) if m==1])
            if bias and hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data = torch.stack([b for b,m in zip(layer.bias.data, mask) if m==1])
            print(len(layer.weight.data),len(layer.bias.data))

        
    def _drop_layer_input(self, layer, mask):
        layer.weight.data = layer.weight.data.transpose(0,1).contiguous()
        self._drop_layer_output(layer, mask, bias=False)
        layer.weight.data = layer.weight.data.transpose(0,1).contiguous()
        print(self)

    def _drop_nodes_layers(self, layer, layers):
        print('dropping nodes from layer %s'%(str(layer)))
        modules = [m for m in layers.modules()][1:]
        mask = self._active_nodes(layer)
        if torch.sum(mask) == 0:
            #this should be edited but right now we cant drop all nodes
            return torch.zeros((), device=mask.device)
        module_ind = modules.index(layer)

        next_ind = module_ind+1
        while not hasattr(modules[next_ind], 'weight'):
            next_ind += 1

        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) or isinstance(layer, misc.BatchNorm):
            #drop batch norm
            self._drop_batch_norm_layer(layer, mask)
            #find previous layer
            module_ind -= 1
            while not hasattr(modules[module_ind], 'weight'):
                module_ind -= 1
            layer = modules[module_ind]
        self._drop_layer_output(layer, mask)
        self._drop_layer_input(modules[next_ind], mask)
        #print('Started: ', layer. ,'Finished: ', )

        return torch.sum(mask)        

    def active_nodes(self):
        raise ValueError('Implemented by Subclass')

    def _active_nodes(self, layer_1):
        b = layer_1.bias
        w = layer_1.weight
        if isinstance(layer_1, torch.nn.modules.batchnorm._BatchNorm) or isinstance(layer_1, misc.BatchNorm):
            #have to handle the case for batch norms differently
            #abs(gamma) * sqrt(m) + b <= 0
            if self.args.batch_norm_func == 'var':
                weight_sum = torch.abs(w) * np.sqrt(w.size(0))
            elif self.args.batch_norm_func == 'l2':
                weight_sum = torch.abs(w)
        else:
            #sum(wx) + b <= 0 with 0 <= x <= 1
            w = w.view((w.size(0), -1))
            weight_sum=torch.sum(torch.max(w,torch.zeros((), device=w.device)), 1)
        value= weight_sum+b
        return value >= 0

    def _active_node_count(self, layer_1):
        count = torch.sum(self._active_nodes(layer_1))
        return count.item()

    def find_layers(self, layers):
        self.first = True
        def _add_module(module, i, modules):
            if self.args.batch_norm:
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, misc.BatchNorm):
                    self.node_drop_layers.append(module)
                    if self.args.l2_lambda_bn:
                        self.reg_layers.append(module)
                else:
                    #if not self.first:
                    self.reg_layers.append(module)
                    self.first = False
            else:
                if not self.first:
                    if isinstance(modules[i+1], ClampedRelu):
                        self.node_drop_layers.append(module)
                self.reg_layers.append(module)
                self.first = False

        self.node_drop_layers = []
        self.reg_layers = []

        modules = [m for m in layers.modules()][1:]
        for i in reversed(range(len(modules))):
            module = modules[i]
            if hasattr(module, 'weight'):
                _add_module(module, i, modules)

        self.node_drop_layers = [layer for layer in reversed(self.node_drop_layers)]
        self.reg_layers = [layer for layer in reversed(self.reg_layers)]

        del self.first

    def epoch_callback(self, epoch):
        if self.args.reg_lambda_anneal is not None:
            if epoch < self.args.reg_lambda_anneal:
                self.reg_lambda = self.args.reg_lambda * (epoch/float(self.args.reg_lambda_anneal))
            else:
                self.reg_lambda = self.args.reg_lambda

        self.init = epoch < self.args.reg_init_epochs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ClassifyConvNetwork(NetworkBase):
    @staticmethod
    def add_args(parser):
        NetworkBase.add_args(parser)
        #convolutional layers
        parser.add_argument('--conv_layer_filters', metavar='N', type=int, nargs='+', default=[32,32,64,64])
        parser.add_argument('--conv_padding', metavar='P', type=int, default=0)
        parser.add_argument('--filter_size', metavar='N', type=int, default=3)
        parser.add_argument('--stride', metavar='N', type=int, default=1)
        parser.add_argument('--max_pool_every', type=int,nargs='+', default=[2], help="MaxPool every this many layers")
        #fully connected layers
        parser.add_argument('--fc_layers', metavar='N', type=int, nargs='+', default=[128,10])
        #all layers
        parser.add_argument('--activ', type=str, default="SoftClamp", help='TrueClamp')
        parser.add_argument('--dropout', type=float, nargs='+', default=[0], help="dropout values, can be a single value applied to all layers or a list of values one for each layer")
        parser.add_argument('--batch_norm', action='store_true', help='use batch norm')
        parser.add_argument('--batch_norm_no_mean_center', action='store_true', help='use batch norm')
        parser.add_argument('--batch_norm_func', type=str, default='var')

        """ Taking out temporarily
        parser.add_argument('--dropout_anneal', action='store_true', help='use dropout and annealing')
        parser.add_argument('--dropout_anneal_p', type=float, default=0.5, help='dropout value')
        parser.add_argument('--dropout_anneal_start', type=int, default=10, help='starting annealing at this epoch')
        parser.add_argument('--dropout_anneal_stop', type=int, default=100, help='stop annealing at this epoch')
        """


    def __init__(self, args, x_size):
        super(ClassifyConvNetwork, self).__init__(args)
        def conv2d_out_size(Hin, layer):
            return (Hin+2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1) / (layer.stride[0])+1

        self.channel_in=x_size[1]
        cur_channel = x_size[1]
        self.batch_size = args.batch_size
        cur_height=x_size[2]
        
        #make max pool into a list if it isnt already
        if len(args.max_pool_every) == 1:
            args.max_pool_every = range(args.max_pool_every[0]-1, len(args.conv_layer_filters), args.max_pool_every[0])
        #make dropout into a list if it isnt already
        if len(args.dropout) == 1:
            args.dropout = [args.dropout[0] for i in range(len(args.conv_layer_filters)+len(args.fc_layers))]

        if (not args.batch_norm_no_mean_center) and args.batch_norm_func == 'var':
            batch_norm_func = nn.BatchNorm2d
        else:
            batch_norm_func = lambda x: misc.BatchNorm(x, mean_center=not args.batch_norm_no_mean_center, norm=args.batch_norm_func)

        layers = []
        for i, cvf in enumerate(args.conv_layer_filters):
            #no bias is batch norm since it includes a bias
            #conv = nn.Conv2d(cur_channel, cvf, kernel_size=args.filter_size, stride=args.stride, padding=args.conv_padding, bias=not args.batch_norm)
            conv = nn.Conv2d(cur_channel, cvf, kernel_size=args.filter_size, stride=args.stride, padding=args.conv_padding)
            layers.append(conv)
            cur_channel = cvf #update cur_channel for next iteration
            #compute the next height so we can make sure the network doent go negative
            next_height = conv2d_out_size(cur_height, layers[-1])
            print("conv: %d %d"%(cur_height,next_height))
            cur_height = next_height
            if next_height <= 0:
                raise ValueError('output height/width is <= 0')
            #batch norm
            if args.batch_norm:
                layer = batch_norm_func(cvf)
                layers.append(layer)
                
            #activations
            layers.append(misc.activation(args.activ))
            #max pool
            if(i in args.max_pool_every):
                cur_height/=2
                layers.append(nn.MaxPool2d(2))
                print(cur_height) 

            #dropout
            dropout = args.dropout[i]
            if dropout > 0:                
                #layers.append(AnnealedDropout(dropout))
                layers.append(nn.Dropout(dropout))

        self.im_size = (cur_channel, cur_height, cur_height)
        self.flat_size = int(cur_channel * cur_height * cur_height)
        self.conv_middle_size= int(cur_height * cur_height)

        layers.append(Flatten())

        if (not args.batch_norm_no_mean_center) and args.batch_norm_func == 'var':
            batch_norm_func = nn.BatchNorm1d
        else:
            batch_norm_func = lambda x: misc.BatchNorm(x, mean_center=not args.batch_norm_no_mean_center, norm=args.batch_norm_func)

        #fully connected layers layers
        cur_size=self.flat_size    
        for i, fc in enumerate(args.fc_layers):
            #no bias if batch norm since it has a bias
            layer = nn.Linear(cur_size, fc)
            layers.append(layer)
            cur_size = fc
            if i != len(args.fc_layers)-1:
                #batch norm
                if args.batch_norm:
                    bn_layer = batch_norm_func(fc)
                    layers.append(bn_layer)
                #activations
                layers.append(misc.activation(args.activ))
                #dropout
                dropout = args.dropout[i+len(args.conv_layer_filters)]
                if dropout > 0:
                    #layers.append(AnnealedDropout(args.all_dropout))
                    layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

        #loss
        self.recon_loss = nn.CrossEntropyLoss()

        #sets up node drop and l2 reg on layers
        self.find_layers(self.layers)

        #initialize network
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

        print(self)

    def forward(self, x):
        return self.layers(x)

    def loss(self, x, y, y_bar):
        recon_loss = self.recon_loss(y_bar, y)
        reg_loss = self.node_drop_loss()
        l2_loss = self.reg_l2_loss()

        acc = self.acc(x,y,y_bar)

        return recon_loss + reg_loss + l2_loss, recon_loss, reg_loss, l2_loss, acc

    def loss_names(self):
        return ['loss', 'op', 'reg', 'l2', 'acc']

    def acc(self, x,y,y_bar):
        predicted = torch.argmax(y_bar, 1)
        acc = torch.mean((predicted == y).float())
        return acc

    def active_nodes(self):
        layer_nodes = [self._active_node_count(layer) for layer in self.node_drop_layers]
        #layer_nodes=[]
        #layer_nodes.extend(self._active_node_layers(self.conv_layers))
        #layer_nodes.extend(self._active_node_layers(self.fc_layers))
        return layer_nodes

        
    def node_drop(self):
        return [self._drop_nodes_layers(layer, self.layers) for layer in self.node_drop_layers]

    """
    def epoch_callback(self, epoch):
        super(ClassifyConvNetwork, self).epoch_callback(epoch)
        if self.args.dropout_anneal:
            if epoch < self.args.dropout_anneal_start:
                self.dropout_p = self.args.dropout_anneal_p
            elif epoch < self.args.dropout_anneal_stop:
                self.dropout_p = self.args.dropout_anneal_p * (1-float(epoch-self.args.dropout_anneal_start)/float(self.args.dropout_anneal_stop-self.args.dropout_anneal_start))
            else:
                self.dropout_p = 0.0
            self._Anneal_Dropouts(conv_layer_filters)
            self._Anneal_Dropouts(fc_layers)
            print("Dropout p: %f"%self.dropout_p)
    def _Anneal_Dropouts(self, layers):
        for i in layers:
            if isinstance(i, AnnealedDropout):
                i.change_p(self.dropout_p)
    """

#**********************************
# VGG
#**********************************
class VGG16(ClassifyConvNetwork):
    @staticmethod
    def add_args(parser):
        ClassifyConvNetwork.add_args(parser)
        parser.add_argument('--vgg_type', metavar='N', type=str, default='cifar_vgg_16', help='type of vgg network')

    def __init__(self, args, x_size):
        type = args.vgg_type
        if type=='16':
            args.conv_layer_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            args.max_pool_every = [1, 3, 6, 9, 12]
            args.fc_layers = [4096, 4096] + args.fc_layers
        elif type=='13':
            args.conv_layer_filters = [64, 64, 128, 128, 256, 256, 512, 512, 512, 512]
            args.max_pool_every = [1, 3, 5, 7, 9]
            args.fc_layers = [4096, 4096] + args.fc_layers
        elif type=='cifar_vgg_16':
            args.conv_layer_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            args.max_pool_every = [1, 3, 6, 9, 12]
            args.fc_layers = [512] + args.fc_layers
            if len(args.dropout) == 1 and args.dropout[0] == 1:
                args.dropout = [0.3, 0.0, 0.4, 0.0, 0.4, 0.4, 0.0, 0.4, 0.4, 0.0, 0.4, 0.4, 0.0, 0.5]
                #args.dropout = [0.3, 0.0, 0.4, 0.0, 0.4, 0.4, 0.0, 0.4, 0.4, 0.0, 0.4, 0.4, 0.0]
        elif type=='16_m':
            args.conv_layer_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]
            args.max_pool_every = [1, 3, 6, 9]
            args.fc_layers = [4096] + args.fc_layers
        elif type=='19':
            args.conv_layer_filters = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512,512, 512, 512, 512,512]
            args.max_pool_every = [1, 3, 7, 11, 15]
            args.fc_layers = [4096, 4096] + args.fc_layers
 
        args.filter_size = 3
        args.conv_padding = 1

        super(VGG16, self).__init__(args, x_size)


#**********************************
# ResNet
#**********************************
class ResidualBottleneck(nn.Module):
    def __init__(self, in_size, out_size, bottleneck, batch_norm=True, stride=1, activ='relu'):
        super(ResidualBottleneck, self).__init__()

        layers = []
        #1
        conv = nn.Conv2d(in_size, bottleneck, kernel_size=1, stride=stride, padding=0)
        layers.append(conv)
        if batch_norm:
            bn = nn.BatchNorm2d(bottleneck)
            layers.append(bn)
        layers.append(misc.activation(activ))
        #2
        conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=3, stride=1, padding=1)
        layers.append(conv)
        if batch_norm:
            bn = nn.BatchNorm2d(bottleneck)
            layers.append(bn)
        layers.append(misc.activation(activ))
        #3
        conv = nn.Conv2d(bottleneck, out_size, kernel_size=1, stride=1, padding=0)
        layers.append(conv)
        if batch_norm:
            bn = nn.BatchNorm2d(out_size)
            layers.append(bn)

        self.layers = nn.Sequential(*layers)

        if in_size != out_size:
            self.downsample = nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, padding=0)
        else:
            self.downsample = None

        self.activ = misc.activation(activ)

    def forward(self, x):
        identity = x

        out = self.layers(x)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity

        return self.activ(out)

class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=True, stride=1, activ='Relu',ksize=3):
        super(ResidualBlock, self).__init__()

        layers = []
        #1
        conv = nn.Conv2d(in_size, out_size, kernel_size=ksize, stride=stride, padding=1)
        layers.append(conv)
        if batch_norm:
            bn = nn.BatchNorm2d(out_size)
            layers.append(bn)
        layers.append(misc.activation(activ))
        #2
        conv = nn.Conv2d(out_size, out_size, kernel_size=ksize, stride=1, padding=1)
        layers.append(conv)
        if batch_norm:
            bn = nn.BatchNorm2d(out_size)
            layers.append(bn)

        self.layers = nn.Sequential(*layers)

        if in_size != out_size:
            self.downsample = nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, padding=0)
        else:
            self.downsample = None

        self.activ = misc.activation(activ)

    def forward(self, x):
        identity = x
        out = self.layers(x)
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        return self.activ(out)


class ResBlock131(ResidualBlock):
    def __init__(self, in_size,mid_size, out_size, batch_norm=True, stride=1, activ='Relu',ksize=3):
        super(ResidualBlock, self).__init__()

        layers = []
        #1
        conv = nn.Conv2d(in_size, mid_size, kernel_size=1, stride=stride, padding=0)
        layers.append(conv)
        if batch_norm:
            bn = nn.BatchNorm2d(mid_size)
            layers.append(bn)
        layers.append(misc.activation(activ))
        #2
        conv = nn.Conv2d(mid_size, mid_size, kernel_size=ksize, stride=1, padding=1)
        layers.append(conv)
        if batch_norm:
            bn = nn.BatchNorm2d(mid_size)
            layers.append(bn)
        layers.append(misc.activation(activ))


        conv = nn.Conv2d(mid_size, out_size, kernel_size=1, stride=1, padding=0)
        layers.append(conv)
        if batch_norm:
            bn = nn.BatchNorm2d(out_size)
            layers.append(bn)



        self.layers = nn.Sequential(*layers)

        if in_size != out_size:
            self.downsample = nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, padding=0)
        else:
            self.downsample = nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, padding=0)

        self.activ = misc.activation(activ)

    def forward(self, x):
        identity = x
        out = self.layers(x)
        #print(identity.shape)
        if self.downsample:
            identity = self.downsample(identity)
        #print(identity.shape,out.shape)
        out += identity
        return self.activ(out)



class ResidualConvNetwork(ClassifyConvNetwork):
    @staticmethod
    def add_args(parser):
        NetworkBase.add_args(parser)
        #convolutional layers
        parser.add_argument('--res_n', metavar='N', type=int, default=6)

        #only used to define the final output size
        parser.add_argument('--fc_layers', metavar='N', type=int, nargs='+', default=[10])

        parser.add_argument('--batch_norm', action='store_true', help='use batch norm')
        parser.add_argument('--batch_norm_func', type=str, default='var')

        parser.add_argument('--activ', type=str, default="SoftClamp", help='TrueClamp')

    def __init__(self, args, x_size):

        NetworkBase.__init__(self, args)

        #only ones supported
        args.batch_norm_func = 'var'

        layers = []

        #initial layer
        cur_sz = x_size[1]
        layers.append(nn.Conv2d(cur_sz, 16, kernel_size=3, stride=1, padding=1))
        if args.batch_norm:
            layers.append(nn.BatchNorm2d(16))
        layers.append(misc.activation(args.activ))
        cur_sz = 16

        for i in range(3):
            sz = 16 * 2**i 
            stride = 1 if i == 0 else 2
            for j in range(args.res_n):
                layers.append(ResidualBlock(cur_sz, sz, batch_norm=args.batch_norm, stride=stride, activ=args.activ))
                stride = 1
                cur_sz = sz

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(Flatten())
        layers.append(nn.Linear(cur_sz, args.fc_layers[-1]))

        self.layers = nn.Sequential(*layers)

        #loss
        self.recon_loss = nn.CrossEntropyLoss()
        
        #sets up node drop and l2 reg on layers
        self.find_layers(self.layers)
        
        
#**********************************
# DenseNet
#**********************************

class ResNet152(ClassifyConvNetwork):
    def add_args(parser):
        NetworkBase.add_args(parser)
        #convolutional layers
        parser.add_argument('--res_n', metavar='N', type=int, default=6)

        #only used to define the final output size
        parser.add_argument('--fc_layers', metavar='N', type=int, nargs='+', default=[10])

        parser.add_argument('--batch_norm', action='store_true', help='use batch norm')
        parser.add_argument('--batch_norm_func', type=str, default='var')

        parser.add_argument('--activ', type=str, default="SoftClamp", help='TrueClamp')

    def __init__(self, args, x_size):

        NetworkBase.__init__(self, args)

        #only ones supported
        args.batch_norm_func = 'var'

        layers = []

        #initial layer
        cur_sz = x_size[1]
        layers.append(nn.Conv2d(cur_sz, 64, kernel_size=7, stride=2, padding=3))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3,stride=2))
        if args.batch_norm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(misc.activation(args.activ))
        cur_sz = 64

        for i in [64]*3+[128]*8+[256]*36+[512]*3:
            sz = i
            stride = 1 if i == 0 else 2
            for j in range(args.res_n):
                layers.append(ResBlock131(cur_sz, sz,4*sz, batch_norm=args.batch_norm, stride=stride, activ=args.activ))
                stride = 1
                cur_sz = 4*sz

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(Flatten())
        layers.append(nn.Linear(cur_sz, args.fc_layers[-1]))

        self.layers = nn.Sequential(*layers)
        print(self)
        #loss
        self.recon_loss = nn.CrossEntropyLoss()

        #sets up node drop and l2 reg on layers
        self.find_layers(self.layers)




class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, k=12, activ='Relu', batch_norm=True):
        super(DenseBasicBlock, self).__init__()
        
        self.conv = nn.Conv2d(inplanes, k, kernel_size=3, padding=1, bias=not batch_norm)
        if batch_norm:
            self.bn = nn.BatchNorm2d(k)
        else:
            self.bn = None
        self.activ = misc.activation(activ)

    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        out = self.activ(out)
        return torch.cat([x, out], dim=1)

class DenseTransitionBlock(nn.Module):
    def __init__(self, inplanes, compression):
        super(DenseTransitionBlock, self).__init__()
        outplanes = int(inplanes / compression)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0)
    def forward(self, x):
        c_out = self.conv(x)
        return F.avg_pool2d(c_out, 2)

class DenseNet(ClassifyConvNetwork):
    @staticmethod
    def add_args(parser):
        NetworkBase.add_args(parser)
        #convolutional layers
        parser.add_argument('--dense_n', metavar='N', type=int, default=12)
        parser.add_argument('--dense_k', metavar='N', type=int, default=12)
        parser.add_argument('--dense_compression', metavar='N', type=float, default=1.0)

        #only used to define the final output size
        parser.add_argument('--fc_layers', metavar='N', type=int, nargs='+', default=[10])

        parser.add_argument('--batch_norm', action='store_true', help='use batch norm')
        parser.add_argument('--batch_norm_func', type=str, default='var')

        parser.add_argument('--activ', type=str, default="SoftClamp", help='TrueClamp')

    def __init__(self, args, x_size):

        NetworkBase.__init__(self, args)

        self.k = args.dense_k

        #only ones supported
        args.batch_norm_func = 'var'

        layers = []
        #initial layer
        cur_sz = x_size[1]
        layers.append(nn.Conv2d(cur_sz, args.dense_k*2, kernel_size=3, stride=1, padding=1))
        if args.batch_norm:
            layers.append(nn.BatchNorm2d(args.dense_k*2))
        layers.append(misc.activation(args.activ))
        cur_sz = args.dense_k*2

        for i in range(args.dense_n):
            layers.append(DenseBasicBlock(cur_sz, args.dense_k, args.activ, args.batch_norm))
            cur_sz += args.dense_k
        
        layers.append(DenseTransitionBlock(cur_sz, args.dense_compression))
        cur_sz = int(cur_sz * args.dense_compression)

        for i in range(args.dense_n):
            layers.append(DenseBasicBlock(cur_sz, args.dense_k, args.activ, args.batch_norm))
            cur_sz += args.dense_k
        
        layers.append(DenseTransitionBlock(cur_sz, args.dense_compression))
        cur_sz = int(cur_sz * args.dense_compression)

        for i in range(args.dense_n):
            layers.append(DenseBasicBlock(cur_sz, args.dense_k, args.activ, args.batch_norm))
            cur_sz += args.dense_k

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(Flatten())
        layers.append(nn.Linear(cur_sz, args.fc_layers[-1]))

        self.layers = nn.Sequential(*layers)

        #loss
        self.recon_loss = nn.CrossEntropyLoss()
        
        #sets up node drop and l2 reg on layers
        self.find_layers(self.layers)

        #print(self.layers)

    def _drop_layer_input_dense(self, layer, mask, dist, k, both=False):
        size = layer.weight.data.size(1)
        #on for all layers after this one
        #then our mask
        #then the remaining ones which may have already been masked off
        mask = torch.cat([torch.ones(size-mask.size(0)-(dist-1)*k, dtype=torch.uint8, device=mask.device), mask, torch.ones((dist-1)*k, dtype=torch.uint8, device=mask.device)])

        layer.weight.data = layer.weight.data.transpose(0,1).contiguous()
        self._drop_layer_output(layer, mask, bias=False)
        layer.weight.data = layer.weight.data.transpose(0,1).contiguous()

        if both:
            self._drop_layer_output(layer, mask, bias=True)

    def _drop_nodes_layers(self, layer, layers):
        print('dropping nodes from layer %s'%(str(layer)))
        modules = [m for m in layers.modules()][1:]
        mask = self._active_nodes(layer)
        module_ind = modules.index(layer)

        next_ind = module_ind+1

        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) or isinstance(layer, misc.BatchNorm):
            #drop batch norm
            self._drop_batch_norm_layer(layer, mask)
            #find previous layer
            module_ind -= 1
            while not hasattr(modules[module_ind], 'weight'):
                module_ind -= 1
            layer = modules[module_ind]

        self._drop_layer_output(layer, mask)

        cur_ind = next_ind
        dist = 1
        while cur_ind < len(modules):
            if hasattr(modules[cur_ind], 'weight'):
                layer = modules[cur_ind]
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) or isinstance(layer, misc.BatchNorm):
                    dist+=1
                else:
                    self._drop_layer_input_dense(layer, mask, dist, self.k, both=isinstance(modules[cur_ind-1], DenseTransitionBlock))
            cur_ind += 1

        return torch.sum(mask)
