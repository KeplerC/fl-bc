#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import vgg16, CNNCifar
from models.Fed import FedAvg
from models.Test import test_img
from utils.util import setup_seed
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import pickle
import _pickle as cPickle
import os
import time


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'exp/fed/{}_{}_{}_C{}_iid{}_{}_user{}_{}'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                                                           args.alpha, args.num_users, current_time)
    # TAG = f'alpha_{alpha}/data_distribution'
    logdir = f'runs/{TAG}' if not args.debug else f'runs2/{TAG}'
    writer = SummaryWriter(logdir)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
    elif args.dataset == 'fmnist':
        dataset_train = datasets.FashionMNIST('../data/fmnist/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize((32, 32)),
                                           transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,)),
                                       ]))

        # testing
        dataset_test = datasets.FashionMNIST('../data/fmnist/', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))
        # test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        dict_users, _ = cifar_noniid(dataset_train, args.num_users, args.alpha)
        for k, v in dict_users.items():
            writer.add_histogram(f'user_{k}/data_distribution',
                                 np.array(dataset_train.targets)[v],
                                 bins=np.arange(11))
            writer.add_histogram(f'all_user/data_distribution',
                                 np.array(dataset_train.targets)[v],
                                 bins=np.arange(11), global_step=k)

    # build model
    if args.model == 'lenet' and (args.dataset == 'cifar' or args.dataset == 'fmnist'):
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'vgg' and args.dataset == 'cifar':
        net_glob = vgg16().to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    test_best_acc = 0.0

    glob_counter = 0
    if not os.path.exists("comm_folder"):
        os.mkdir("comm_folder")
    for iter in range(args.epochs):
        # w_locals, loss_locals = [], []
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(10), m, replace=False)
        idxs_users = list(range(2))
        if glob_counter == 21:
            break
        
        # load net_glob
        glob_file_name = "comm_folder/" +  "glob" + str(glob_counter)
        if glob_counter != 0:
            while not os.path.exists(glob_file_name):
                time.sleep(1)
            if os.path.isfile(glob_file_name):
                with open(glob_file_name, "rb") as input_file:
                    net_glob = cPickle.load(input_file)
                    print(glob_file_name)
            else:
                raise ValueError("%s isn't a file!" % glob_file_name)
        
        local_counter = 0
            
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[0])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

        file_name_w = "comm_folder/" + "w" + str(glob_counter) + "_" + str(local_counter)
        file_name_loss = "comm_folder/" + "loss" + str(glob_counter) + "_" + str(local_counter)
        with open(file_name_w, "wb") as output_file:
            cPickle.dump(w, output_file)
        with open(file_name_loss, "wb") as output_file:
            cPickle.dump(loss, output_file)
        # w_locals.append(w)
        # loss_locals.append(loss)

        
        glob_counter += 1
            
        # update global weights
        # w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        # net_glob.load_state_dict(w_glob)

        # print loss
        # loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Train loss {:.3f}'.format(iter, loss_avg))
        # loss_train.append(loss_avg)
        # writer.add_scalar('train_loss', loss_avg, iter)
        # test_acc, test_loss = test_img(net_glob, dataset_test, args)
        # writer.add_scalar('test_loss', test_loss, iter)
        # writer.add_scalar('test_acc', test_acc, iter)

        # save_info = {
        #    "model": net_glob.state_dict(),
        #    "epoch": iter
        # }
        # save model weights
        # if (iter+1) % 500 == 0:
        #    save_path = f'./save2/{TAG}_{iter+1}es' if args.debug else f'./save/{TAG}_{iter+1}es'
        #    torch.save(save_info, save_path)
        #if iter > 100 and test_acc > test_best_acc:
        #    test_best_acc = test_acc
        #    save_path = f'./save2/{TAG}_bst' if args.debug else f'./save/{TAG}_bst'
        #    torch.save(save_info, save_path)

    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))
    # writer.close()
