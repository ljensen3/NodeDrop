# NodeDrop
This repo contains the code for the paper "NodeDrop: A Method for Finding Sufficient Network Architecture Size with Improved Generalization". We define a condition for dropping nodes in a network and then use L1 regularization to bias nodes toward this condition.

requirements This repo requires the following non-standard python packages argparse torch - 1.0 torch-vision - 1.0 progress

Cifar Experements
To run the CIFAR runs use the python script run_cifar.py. For example to run vgg16 on cifar 10 with batch normalization run python run_cifar.py <gpu_id> vgg_cifar10_bn to test on the same runs run python run_cifar.py <gpu_id> vgg_cifar10_bn test
