import sys

import node_drop

param_dict = {}

param_dict["resume"]=("", "--resume")

param_dict["vgg"]=("vgg", "", "vgg")
param_dict["dense40"]=("dense40", "--dense_k 12 --dense_n 12", "dense")

param_dict["mnist"]=("mnist","--lr 1e-3 --fc_layers 10 --epochs 10 --reg_lambda 0.0")
param_dict["cifar10"]=("cifar", "--dset CIFAR --lr 1e-3 --fc_layers 10 --epochs 200 --reg_lambda 0.0 --batch_size 128 --csv_file cifar10")
param_dict["cifar100"]=("cifar100", "--dset CIFAR100 --lr 1e-3 --fc_layers 100 --epochs 200 --reg_lambda 0.0 --batch_size 128 --csv_file cifar100")

param_dict["relu"]=("relu", "--activ Relu")

param_dict["ep100"]=("ep100", "--epochs 100")
param_dict["ep400"]=("ep400", "--epochs 400")
param_dict["ep600"]=("ep600", "--epochs 600")
param_dict["ep700"]=("ep700", "--epochs 700")
param_dict["ep700"]=("ep700", "--epochs 700")
param_dict["ep2000"]=("ep2000", "--epochs 2000")

param_dict["sgd"]=("sgd", "--sgd --lr_step 80 130 --lr 0.1 --batch_size 128")
param_dict["sgdDe"]=("sgd", "--sgd --lr_step 150 225 --lr 0.1 --batch_size 128 --epochs 300")
param_dict["adam"]=("adam", "--lr 1e-3 --batch_size 512")

param_dict["l21e6"]=("l21e6", "--l2_lambda 1e-6")
param_dict["l21e5"]=("l21e5", "--l2_lambda 1e-5")
param_dict["l25e4"]=("l25e4", "--l2_lambda 5e-4")
param_dict["l21e4"]=("l21e4", "--l2_lambda 1e-4")
param_dict["l21e3"]=("l21e3", "--l2_lambda 1e-3")
param_dict["l21e2"]=("l21e2", "--l2_lambda 1e-2")

#batch norms
param_dict["bn"]=("bn", "--batch_norm --activ Relu")

#regularization
param_dict["ie140"]=("ie140", "--reg_init_epochs 140")
param_dict["ie100"]=("ie100", "--reg_init_epochs 100")
param_dict["ie50"]=("ie50", "--reg_init_epochs 50")
param_dict["ie20"]=("ie20", "--reg_init_epochs 20")
param_dict["ie5"]=("ie5", "--reg_init_epochs 5")

param_dict["reg1"]=("reg1", "--reg_lambda 1e-1 --reg_C 10.0")
param_dict["reg2"]=("reg2", "--reg_lambda 1e-2 --reg_C 10.0")
param_dict["reg3"]=("reg3", "--reg_lambda 1e-3 --reg_C 10.0")
param_dict["reg3e4"]=("reg3e4", "--reg_lambda 3.162e-4 --reg_C 10.0")
param_dict["reg4"]=("reg4", "--reg_lambda 1e-4 --reg_C 10.0")
param_dict["reg3e5"]=("reg3e5", "--reg_lambda 3.162e-5 --reg_C 10.0")
param_dict["reg5e5"]=("reg5e5", "--reg_lambda 5e-5 --reg_C 10.0")
param_dict["reg5"]=("reg5", "--reg_lambda 1e-5 --reg_C 10.0")
param_dict["reg3e6"]=("reg3e6", "--reg_lambda 3.162e-6 --reg_C 10.0")
param_dict["reg6"]=("reg6", "--reg_lambda 1e-6 --reg_C 10.0")
param_dict["reg3e7"]=("reg3e7", "--reg_lambda 3.162e-7 --reg_C 10.0")
param_dict["reg7"]=("reg7", "--reg_lambda 1e-7 --reg_C 10.0")
param_dict["reg8"]=("reg8", "--reg_lambda 1e-8 --reg_C 10.0")

param_dict["data_parallel"]=("","--data_parallel 4 5 6 7")


def run(params, net='vgg'):
    names=[]
    args=[]
    
    for param in params:
        ret = param_dict[param]
        if len(ret) == 3:
            name, arg, net = ret
        else:
            name, arg = ret
        if name != "":
            names.append(name)
        if arg:
            args += arg.split(" ")

    if len(sys.argv)>=4 and sys.argv[3] == 'test':
        args.append('--test')

    node_drop.main(['_'.join(names), net, sys.argv[1]] + args)


#run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4",'data_parallel'])
if sys.argv[2] == 'vgg_cifar10_bn':
    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4"])

    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4", "reg7", "ep400", "ie5"])
    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4", "reg3e7", "ep400", "ie5"])
    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4", "reg6", "ep400", "ie5"])
    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4", "reg3e6", "ep400", "ie5"])
    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4", "reg5", "ep400", "ie5"])
    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4", "reg3e5", "ep400", "ie5"])
    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4", "reg4", "ep400", "ie5"])
    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4", "reg3e4", "ep400", "ie5"])
    run(["vgg", "cifar10", "bn", "relu", "sgd", "l25e4", "reg3", "ep400", "ie5"])

if sys.argv[2] == 'vgg_cifar100_nbn':
    run(["vgg", "cifar10", "adam", "l25e4"])

    run(["vgg", "cifar10", "adam", "l25e4", "reg7", "ep400", "ie5"])
    run(["vgg", "cifar10", "adam", "l25e4", "reg3e7", "ep400", "ie5"])
    run(["vgg", "cifar10", "adam", "l25e4", "reg6", "ep400", "ie5"])
    run(["vgg", "cifar10", "adam", "l25e4", "reg3e6", "ep400", "ie5"])
    run(["vgg", "cifar10", "adam", "l25e4", "reg5", "ep400", "ie5"])
    run(["vgg", "cifar10", "adam", "l25e4", "reg3e5", "ep400", "ie5"])
    run(["vgg", "cifar10", "adam", "l25e4", "reg4", "ep400", "ie5"])
    run(["vgg", "cifar10", "adam", "l25e4", "reg3e4", "ep400", "ie5"])
    run(["vgg", "cifar10", "adam", "l25e4", "reg3", "ep400", "ie5"])

if sys.argv[2] == 'vgg_cifar100':
    #def batch_norm
    run(["vgg", "cifar100", "bn", "relu","sgd", "l25e4"])
    #res batch_norm
    run(["vgg", "cifar100", "bn", "relu", "sgd", "l25e4", "reg6", "ep400"])
    run(["vgg", "cifar100", "bn", "relu", "sgd", "l25e4", "reg5", "ep400"])
    run(["vgg", "cifar100", "bn", "relu", "sgd", "l25e4", "reg4", "ep400"])


if sys.argv[2] == 'dense40_cifar10_bn':
    #def batch_norm
    run(["dense40", "cifar10", "bn", "relu","sgdDe", "l21e4"])
    #res batch_norm
    run(["dense40", "cifar10", "bn", "relu", "sgdDe", "l21e4", "reg6", "ep600"])
    run(["dense40", "cifar10", "bn", "relu", "sgdDe", "l21e4", "reg5", "ep600"])
    run(["dense40", "cifar10", "bn", "relu", "sgdDe", "l21e4", "reg4", "ep600"])

if sys.argv[2] == 'dense40_cifar10_nbn':
    run(["dense40", "cifar10", "adam"])
    run(["dense40", "cifar10", "adam", "reg6", "ep600"])
    run(["dense40", "cifar10", "adam", "reg5", "ep600"])
    run(["dense40", "cifar10", "adam", "reg4", "ep600"])

if sys.argv[2] == 'dense40_cifar100':
    #def batch_norm
    run(["dense40", "cifar100", "bn", "relu","sgdDe", "l21e4"])
    #res batch_norm
    run(["dense40", "cifar100", "bn", "relu", "sgdDe", "l21e4", "reg6", "ep600"])
    run(["dense40", "cifar100", "bn", "relu", "sgdDe", "l21e4", "reg5", "ep600"])
    run(["dense40", "cifar100", "bn", "relu", "sgdDe", "l21e4", "reg4", "ep600"])
