'''Experimenting training from scratch'''

'''This is used for standard domain generalization'''
import torchvision
from utils import *
import numpy as np

import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os, socket, random
import config

from models.preactresnet import PreActResNet18_encoder, VAE_Small, FDC_deep_preact
from models.resnet import FDC5

from torchvision.utils import save_image



def loss_function(recon_x, x, mu, logvar, beta):
    BCE = F.binary_cross_entropy(recon_x.view(x.size(0), -1), x.view(x.size(0), -1), reduction='mean')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD * beta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--eval-batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=5e-5, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--workers', default=50, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--samples', default=10, type=int)
    parser.add_argument('--ssl_epoch', default=500, type=int)
    parser.add_argument('--fd_epoch', default=10, type=int) ###############
    parser.add_argument('--modelname', default='res50-4x', type=str) ##############
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='SimCLR_FD_new', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--drop_xp', action='store_true')
    parser.add_argument('--drop_xp_ratio', default=0.5, type=float)
    parser.add_argument('--scale', default=0.01, type=float)
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--noise_inside', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--shorter', action='store_true')
    parser.add_argument('--half_eval', action='store_true') # Only train half epoch and do evaluation, so we can do more finegrained early stop.
    parser.add_argument('--train_ssl', action='store_true')
    parser.add_argument('--all_65', action='store_true')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--dtest', default='None', type=str)
    parser.add_argument('--style_test', default='', type=str, choices=['D1', 'D2', 'D3'])
    parser.add_argument('--chkpt-iters', default=10, type=int)
    if 'cv' in socket.gethostname():
        parser.add_argument('--save_root_path', default='/proj/vondrick/mcz/FrontDoor/NewOurs/', type=str)
    else:
        parser.add_argument('--save_root_path', default='/local/rcs/mcz/2021Spring/FrontDoor/', type=str)
    return parser.parse_args()

def main():
    eval_fd = False

    args = get_args()
    import uuid
    import datetime
    unique_str = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    args.fname = os.path.join(args.save_root_path, args.fname, timestamp + unique_str)
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    
    # Specify SSL Model
    latent_dim = 8192 # The Following Res50x4 is 8192 out dim
    outdim = 1000

    from models.resnet_wider_simclr import resnet50x4, resnet50x2, resnet50x1

    if args.modelname == 'res50-4x':
        resnet = resnet50x4()
        sd = '/proj/vondrick2/chengzhi/ssl_pretrained/simclr-converter/resnet50-4x.pth'
        latent_dim = 8192  # The Following Res50x4 is 8192 out dim
    elif args.modelname == 'res50-2x':
        resnet = resnet50x2()
        sd = '/proj/vondrick2/chengzhi/ssl_pretrained/simclr-converter/resnet50-2x.pth'
        latent_dim = 4096  # The Following Res50x4 is 8192 out dim
    elif args.modelname == 'res50-1x':
        resnet = resnet50x1()
        sd = '/proj/vondrick2/chengzhi/ssl_pretrained/simclr-converter/resnet50-1x.pth'
        latent_dim = 2048

    sd = torch.load(sd, map_location='cpu')
    resnet.load_state_dict(sd['state_dict'])

    resnet = nn.DataParallel(resnet).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    
    
    

    from dataloader.multidomain_loader import DomainTest, RandomData, MultiDomainLoader
    if socket.gethostname()=='cv10':
        root_path="/local/vondrick/chengzhi/ImageNet-Data"
    elif socket.gethostname() == 'cv02':
        root_path = "/proj/vondrick/mcz/ImageNet-Data"
    elif 'cv' in socket.gethostname():
        root_path = "/proj/vondrick/mcz/ImageNet-Data"
    sketch_root = '/proj/vondrick2/datasets/ImageNet-OOD/sketch'
    redition_root = '/proj/vondrick2/datasets/ImageNet-OOD/imagenet-redition'
    fore_back_root = '/proj/vondrick/james/bg_challenge_prod'

    if socket.gethostname() == 'cv02' or socket.gethostname() == 'cv04':
        sketch_root = '/local/vondrick/chengzhi/sketch'
        redition_root = '/local/vondrick/chengzhi/imagenet-redition'


    test_d = args.style_test
    train_sampler = None

    if not eval_fd:
        train_dataset = MultiDomainLoader(dataset_root_dir=root_path,
                                                train_split=['train'], subsample=1, noNormalize=True)  # , 'D2'
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_data_sketch = DomainTest(dataset_root_dir=sketch_root,
                           test_split=['val'], noNormalize=True) # As the download model is not using normalized input, we need to set noNormalize to True
    test_data_redition = DomainTest(dataset_root_dir=redition_root,
                                  test_split=['val'],
                                  noNormalize=True)  # As the download model is not using normalized input, we need to set noNormalize to True
    test_rand_data = RandomData(dataset_root_dir=root_path,
                                all_split=['train'], noNormalize=True) # As the download model is not using normalized input, we need to set noNormalize to True

    print('datapath', root_path)

    test_loader_sketch = torch.utils.data.DataLoader(
        test_data_sketch, batch_size=args.eval_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader_redition = torch.utils.data.DataLoader(
        test_data_redition, batch_size=args.eval_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    random_loader = torch.utils.data.DataLoader(
        test_rand_data, batch_size=args.eval_batch_size*args.samples, shuffle=True,
        num_workers=args.workers*2, pin_memory=True, sampler=train_sampler)

    
    # # NOTICE, the original model do not have normalization
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    #
    valdir = '/proj/vondrick2/datasets/ImageNet-OOD/imagenet-redition/val'
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(sketch_root, 'val'), transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # FD Classifier
    print('fdc5')
    classifier = FDC5(hidden_dim=latent_dim, cat_num=outdim, drop_xp=args.drop_xp, drop_xp_ratio=args.drop_xp_ratio).cuda()
    classifier = nn.DataParallel(classifier)
    classifier.train()


    if eval_fd:
        # sd = torch.load('/proj/vondrick/mcz/FrontDoor/NewOurs/SimsaveforCam/v1/model_best_sk.pth', map_location='cpu')
        sd = torch.load('/proj/vondrick/mcz/FrontDoor/NewOurs/SimsaveforCam/v1/model_best_r.pth', map_location='cpu')
        classifier.load_state_dict(sd['state_dict_classifier'])
        print('acc', sd['test_robust_acc'])

    # Optimizer to Train P(Y|z, x)
    params = list(classifier.parameters())  # list(resnet.parameters()) +
    opt = torch.optim.Adam(params, lr=1e-3)

    r_best_test_robust_acc = 0
    st_best_test_robust_acc = 0
    sk_best_test_robust_acc = 0
    best_val_robust_acc = 0

    start_epoch = 1

    epochs = args.fd_epoch
    print(epochs, 'FD5')

    torch.cuda.empty_cache()

    train_acc_list=[]
    testfd_acc_list=[]
    testvani_acc_list=[]

    scale = args.scale



    for epoch in range(start_epoch, epochs+1):

        resnet.eval()
        classifier.train()

        start_time = time.time()
        train_loss = 0
        train_acc = 0

        train_n = 0

        # TRaining

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        top1ori = AverageMeter('Acc@1 ori', ':6.2f')
        top5ori = AverageMeter('Acc@5 ori', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5, top1ori, top5ori],
            prefix="Epoch: [{}]".format(epoch))

        for i, batch in enumerate(train_loader):
            if i==3 and args.shorter:
                break
            if i==len(train_loader)//2 and args.half_eval:
                break
            if args.eval:
                break
            X, Xp, y = batch
            X = X.cuda()
            Xp = Xp.cuda()  # TODO: just fix this bug, previous use X.
            y = y.cuda()

            with torch.no_grad():
                original_out, fea = resnet(X)
                fea = fea.detach()  #
                if not args.noise_inside: # if don't add individual in FDC forward pass, then do it here, which is a universal noise.
                    fea = fea + scale * torch.normal(mean=torch.zeros_like(fea), std=torch.ones_like(fea))

            flag=True # detach always

            prediction = classifier(fea, Xp, False, random_detach=flag, noise_inside=args.noise_inside) # always detach

            cl_loss = criterion(prediction, y)
            loss = cl_loss

            train_loss += loss.item() * y.size(0)
            train_acc += (prediction.max(1)[1] == y).sum().item()

            losses.update(loss.item(), X.size(0))

            acc1, acc5 = accuracy(prediction, y, topk=(1, 5))
            acc1ori, acc5ori = accuracy(original_out, y, topk=(1, 5))
            top1.update(acc1[0], X.size(0))
            top5.update(acc5[0], X.size(0))
            top1ori.update(acc1ori[0], X.size(0))
            top5ori.update(acc5ori[0], X.size(0))

            train_n += y.size(0)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 500 == 0:
                progress.display(i)

        print('start eval')
        train_time = time.time()


        def testing(test_loader, infostr):

            classifier.eval()

            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0

            ##############################
            # Causal Test
            batch_time = AverageMeter('Time', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(test_loader),
                [batch_time, losses, top1, top5],
                prefix=f'{infostr} Causal Test: ')


            with torch.no_grad():
                for i, bs_pair in enumerate(zip(test_loader, random_loader)):
                    batch, batch_rand = bs_pair
                    X, y = batch
                    Xp, _ = batch_rand

                    X = X.cuda()
                    y = y.cuda()
                    Xp = Xp.cuda()

                    out, fea = resnet(X)
                    if not args.noise_inside:
                        feature = fea + scale * torch.normal(mean=torch.zeros_like(fea), std=torch.ones_like(fea))
                    else:
                        feature = fea

                    # TODO: x_pair
                    bs_m = feature.size(0)
                    j=0
                    logit_compose = classifier(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :], noise_inside=args.noise_inside)

                    for jj in range(args.samples-1):
                        if not args.noise_inside:
                            feature = fea + scale * torch.normal(mean=torch.zeros_like(fea), std=torch.ones_like(fea))
                        else:
                            feature = fea
                        logit_compose = logit_compose + classifier(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :],
                                                                    noise_inside=args.noise_inside)  # TODO:

                    if 'redition' in infostr:
                        logit_compose = logit_compose[:, config.imagenet_r_mask]

                    test_robust_acc += (logit_compose.max(1)[1] == y).sum().item()
                    test_robust_loss += loss.item() * y.size(0)
                    test_n += y.size(0)

                    acc1, acc5 = accuracy(logit_compose, y, topk=(1, 5))
                    top1.update(acc1[0], X.size(0))
                    top5.update(acc5[0], X.size(0))

                    if i>1 and args.fast:
                        break
                    if i % 200 == 0:
                        progress.display(i)
            ##############################

            train_time = time.time()

            classifier.eval()

            ##############################
            # Baseline Test
            test_vanilla_acc = 0
            test_vanilla_loss=0
            test_n_v = 0

            batch_time = AverageMeter('Time', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1_bl = AverageMeter('Acc@1', ':6.2f')
            top5_bl = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(test_loader),
                [batch_time, losses, top1_bl, top5_bl],
                prefix=f'{infostr} Spurious Test: ')


            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    X, y = batch

                    X = X.cuda()
                    y = y.cuda()

                    out, fea = resnet(X)
                    if not args.noise_inside:
                        feature = fea + scale * torch.normal(mean=torch.zeros_like(fea), std=torch.ones_like(fea))
                    else:
                        feature = fea

                    # TODO: x_pair
                    bs_m = feature.size(0)
                    j = 0
                    logit_compose = classifier(feature, X, test=False, noise_inside=args.noise_inside)

                    if 'redition' in infostr:
                        logit_compose = logit_compose[:, config.imagenet_r_mask]

                    test_vanilla_acc += (logit_compose.max(1)[1] == y).sum().item()
                    test_vanilla_loss += loss.item() * y.size(0)
                    test_n_v += y.size(0)

                    if i>1 and args.fast:
                        break

                    acc1, acc5 = accuracy(logit_compose, y, topk=(1, 5))
                    top1_bl.update(acc1[0], X.size(0))
                    top5_bl.update(acc5[0], X.size(0))

                    # if i>4 and args.fast:
                    #     break
                    if i % 200 == 0:
                        progress.display(i)


            test_time = time.time()
            ##############################

            print('\n\n', infostr, "epoch", epoch, "test domain", test_d, " train acc", train_acc / train_n,
                    "test baseline Accuracy:", top1_bl.avg, "test causal Accuracy:", top1.avg)

            return train_acc, top1.avg, top1_bl.avg

        train_acc, test_robust_acc_r, test_vanilla_acc_r = testing(test_loader_redition, 'redition')
        train_acc, test_robust_acc_sk, test_vanilla_acc_sk = testing(test_loader_sketch, 'Sketch')

        torch.save({'classifier': classifier.state_dict()},
                    os.path.join(args.fname, f'model_{epoch}.pth'))
        torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))

        # save best
        if test_robust_acc_r > r_best_test_robust_acc:
            torch.save({
                'state_dict_classifier': classifier.state_dict(),
                'state_dict_resnet': resnet.state_dict(),
                'test_robust_acc': test_robust_acc_r,
                'test_acc': test_vanilla_acc_r,
                'epoch': epoch,
            }, os.path.join(args.fname, f'model_best_r.pth'))
            r_best_test_robust_acc = test_robust_acc_r

        # save best
        if test_robust_acc_sk > sk_best_test_robust_acc:
            torch.save({
                'state_dict_classifier': classifier.state_dict(),
                'state_dict_resnet': resnet.state_dict(),
                'test_robust_acc': test_robust_acc_sk,
                'test_acc': test_vanilla_acc_sk,
                'epoch': epoch,
            }, os.path.join(args.fname, f'model_best_sk.pth'))
            sk_best_test_robust_acc = test_robust_acc_sk

    print(train_acc_list)
    print(testfd_acc_list)
    print(testvani_acc_list)


if __name__ == "__main__":
    main()