import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from plotly import tools
import plotly.offline as of
import plotly.graph_objs as go

class bcolors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FAIL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def activation(activ):
    if(activ=='TrueClamp'):
        return TrueClampedRelu()
    elif(activ=='SoftClamp'):
        return ClampedRelu(10.0)
    else:
        return nn.ReLU(inplace=True)


class ClampedRelu(nn.Module):
    def __init__(self, beta=10.0):
        super(ClampedRelu, self).__init__()
        self.sp = nn.Softplus(beta=beta)
        #self.shift = -1 + self.sp(1)
        #self.mult = 1./(1.+shift)
        #self.scaled_sp = sp 

    def forward(self, x):
        #return F.relu(self.mult*(1 - self.sp(1-x)+self.shift))
        return F.relu(1 - self.sp(1-x))

class TrueClampedRelu(nn.Module):
    def __init__(self):
        super(TrueClampedRelu, self).__init__()
        
    def forward(self, x):
        return F.relu(1 - F.relu(1-x))

class AnnealedDropout(nn.Module):
    def __init__(self, p=0.0):
        super(AnnealedDropout, self).__init__()
        self.p=p
    
    def change_p(p=0.0):
        self.p=p
        print(self.p)
    
    def forward(self, x):
        return F.dropout(x,self.p)

class StraightThroughRelu(nn.Module):
    def __init__(self, st_grad=0.0, softplus=None):
        super(StraightThroughRelu, self).__init__()
        self.relu = nn.ReLU()
        self.register_buffer('st', torch.tensor(st_grad))
        self.softplus = softplus
        
        if softplus:
            def hook(module, inp, out):
                self.inp = inp[0]
            self.relu.register_forward_hook(hook)

        def hook(module, grad_input, grad_output):
            
            mask = torch.abs(grad_input[0]) > 0
            if self.softplus:
                beta = 2
                e_beta = torch.exp(beta * (self.inp+1))
                grad = e_beta / (1 + e_beta)
                out = (torch.where(mask,grad_input[0], self.st * grad * grad_output[0]),)
            else:
                out = (torch.where(mask,grad_input[0], self.st * grad_output[0]),)
            return out

        if st_grad != 0.0:
            self.relu.register_backward_hook(hook)

    def forward(self, x):
        return self.relu(x)


class StraightThroughFC(nn.Module):
    def __init__(self, in_features, out_features, st_grad=0.0):
        super(StraightThroughFC, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features, out_features)
        self.register_buffer('st', torch.tensor(st_grad))

        def relu_hook(module, grad_input, grad_output):
            self.mask = torch.abs(grad_input[0]) > 0
            out = torch.where(self.mask, grad_input[0], self.st * grad_output[0])
            return (out,)

        def fc_hook(module, grad_input, grad_output):
            #mask off added grads  
            grad_output_m = grad_output[0] * self.mask.float()
            return (grad_input[0], torch.matmul(grad_output_m, self.fc.weight), grad_input[2])

        if st_grad != 0.0:
            self.relu.register_backward_hook(relu_hook)
            self.fc.register_backward_hook(fc_hook)

    def forward(self, x):
        return self.relu(self.fc(x))
        

def to_image(x, fig=None, ax=None, outdir=None):
    save=False
    if fig is None:
        save=True
        fig = plt.figure(figsize=(20, 4))
    if ax is None:
        ax = fig.gca()
    if isinstance(x, np.ndarray):
        x = np.reshape(np.clip(x, 0.0, 1.0), (28,28))
    else:
        x = torch.clamp(x,0.0,1.0).view(28,28).detach().cpu().numpy()
    ax.imshow(x, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if save:
        plt.gray() #grey scale

        plt.tight_layout(pad=1, h_pad=3.0)
        if outdir:
            outfile = outdir+'implt.png'
        else:
            outfile = './plts/implt.png'
        print('saving plot %s'%outfile)
        plt.savefig(outfile)

def batch_to_image(x, x_hat, outdir=None):
    fig = plt.figure(figsize=(20, 4))
    for i in range(10):
        ax = plt.subplot(2,10,i+1)
        to_image(x[i],fig,ax)
        ax = plt.subplot(2,10,10+i+1)
        to_image(x_hat[i],fig,ax)

    
    plt.gray() #grey scale
    plt.tight_layout(pad=1, h_pad=3.0)
    if outdir:
        outfile = outdir+'/implt.png'
    else:
        outfile = './plts/implt.png'
    print('saving plot %s'%outfile)
    plt.savefig(outfile)

def sweep_to_image(x, outdir=None):
    n = len(x)
    fig = plt.figure(figsize=(2*n, 20))
    for i, x_ in enumerate(x):
        for j in range(10):
            ax = plt.subplot(10,n,i+j*n+1)
            to_image(x[i][j],fig,ax)

    plt.gray() #grey scale
    plt.tight_layout(pad=1, h_pad=3.0)
    if outdir:
        outfile = outdir+'/implt.png'
    else:
        outfile = './plts/implt.png'
    print('saving plot %s'%outfile)
    plt.savefig(outfile)

def plot_sweep(ckpt_dir, lambda_l, op_loss_l, nodes_l):
    fig = tools.make_subplots(rows=2, cols=2)

    trace = go.Scatter(
        x=lambda_l,
        y=op_loss_l,
        name='Lambda vs Loss',
        mode = 'markers'
    )

    fig.append_trace(trace, 1, 1)
    fig['layout']['xaxis1'].update(title='Lambda*C', type='log')
    fig['layout']['yaxis1'].update(title='Loss')
    
    trace = go.Scatter(
        x=lambda_l,
        y=nodes_l,
        name='Lambda vs Dim',
        mode = 'markers'
    )
    
    fig.append_trace(trace, 1, 2)
    fig['layout']['xaxis2'].update(title='Lambda*C', type='log')
    fig['layout']['yaxis2'].update(title='Dim')

    dim = -1
    dims = []
    loss = []
    for d, l in zip(nodes_l, op_loss_l):
        if d != dim:
            dims.append(d)
            loss.append(l)
            dim = d

    trace = go.Scatter(
        x=nodes_l,
        y=op_loss_l,
        name='Dim vs Loss',
        mode = 'markers'
    )

    fig.append_trace(trace, 2, 1)
    fig['layout']['xaxis3'].update(title='Dim')
    fig['layout']['yaxis3'].update(title='Loss')

    outfile = ckpt_dir + '/plot.html'
    fig['layout'].update(height=800, width=1000)
    of.plot(fig, filename=outfile, auto_open=False)
    print('Created Plot %s'%outfile)
        
#Coppied from nn._BatchNorm Version 2
#updated for extra cases
# 1 - no 1/m in variance, just divide by l2_norm of mean centered
# 2 - l1 norm
# 3 - no mean centered
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, mean_center=True, norm='var'):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if norm == 'l2' or norm =='l1':
            print('turning track_running_stats off thanks to norm settings')
            self.track_running_stats = False
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            self.register_buffer('init_running', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        #added
        self.mean_center = mean_center
        self.norm = norm
        if norm == 'var':
            self.norm_func = lambda x, dims: torch.sqrt(torch.sum(x**2, dim=dims)/x.size(0) + self.eps)
        elif norm == 'l2':
            self.norm_func = lambda x, dims: torch.sqrt(torch.sum(x**2, dim=dims) + self.eps)
        elif norm == 'l1':
            self.norm_func = lambda x, dims: torch.sum(torch.abs(x), dim=dims) + self.eps

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        return True

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        comp_avg = self.training or not self.track_running_stats

        if input.dim() == 2:
            dims = 0
            view = [1, -1]
        elif input.dim() == 3:
            dims = [0,2]
            view = [1, -1, 1]
        elif input.dim() == 4:
            dims = [0,2,3]
            view = [1, -1, 1, 1]

        if not comp_avg:
            mean = self.running_mean
            var = self.running_var
            imm = input - mean.view(view)
        else:
            mean = torch.mean(input, dim=dims)
            imm = input - mean.view(view)
            var = self.norm_func(imm, dims)
            if self.track_running_stats:
                if self.init_running == 0:
                    self.running_mean[:] = mean
                    self.running_var[:] = var
                    self.init_running.add_(1)
                else:
                    self.running_mean[:] = exponential_average_factor * mean + (1-exponential_average_factor) * self.running_mean
                    self.running_var[:] = exponential_average_factor * var + (1-exponential_average_factor) * self.running_var

        if self.mean_center:
            a_hat = imm / var.view(view)
        else:
            a_hat = input / var.view(view)

        if self.weight is not None:
            return self.weight.view(view) * a_hat + self.bias.view(view)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}, mean_center={mean_center}, norm={norm}'.format(**self.__dict__)
