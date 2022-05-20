'''Experimenting training from scratch'''
'''This is the code used in our paper for waterbird experiment'''
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

# from models.preactresnet import PreActResNet18_encoder, VAE_Small, FDC_deep_preact

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
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--eval-batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=5e-5, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lambda_vae', default=1, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--samples', default=20, type=int)
    parser.add_argument('--ssl_epoch', default=500, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='waterbird2-cvpr-reproduce', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--subsample', default=1, type=int)
    parser.add_argument('--middle_hidden', default=1024, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--drop_xp', action='store_true')
    parser.add_argument('--drop_xp_ratio', default=0.5, type=float)
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--shorter', action='store_true')
    parser.add_argument('--train_ssl', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--dtest', default='None', type=str)
    parser.add_argument('--chkpt-iters', default=10, type=int)
    if 'cv' in socket.gethostname():
        parser.add_argument('--save_root_path', default='./', type=str)
    else:
        parser.add_argument('--save_root_path', default='/local/rcs/mcz/2021Spring/FrontDoor/', type=str)
    return parser.parse_args()

def main():
    adda_times=1

    args = get_args()
    import uuid
    import datetime
    unique_str = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    args.fname = os.path.join(args.save_root_path, args.fname, timestamp + unique_str)
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    from dataloader.waterbird_loader import WB_DomainTest, WB_RandomData, WB_MultiDomainLoaderTripleFD

    root_path = "/proj/vondrick2/datasets/ImageNet-OOD/waterbird_complete95_forest2water2"
    if socket.gethostname()=='cv02':
        root_path = "/local/vondrick/cz/waterbird_complete95_forest2water2"
    elif socket.gethostname() == 'cv08':
        root_path = "/local/vondrick/chengzhi/waterbird_complete95_forest2water2"

    outdim=2

    train_dataset = WB_MultiDomainLoaderTripleFD(dataset_root_dir=root_path,
                                            train_split='train')  # , 'D2'
    val_data_list = [WB_DomainTest(dataset_root_dir=root_path, split='val', group=[0,0]),
                WB_DomainTest(dataset_root_dir=root_path, split='val', group=[0, 1]),
                WB_DomainTest(dataset_root_dir=root_path, split='val', group=[1, 0]),
                WB_DomainTest(dataset_root_dir=root_path, split='val', group=[1, 1])]
    test_data_list = [WB_DomainTest(dataset_root_dir=root_path,split='test', group=[0,0]),
                 WB_DomainTest(dataset_root_dir=root_path, split='test', group=[0, 1]),
                 WB_DomainTest(dataset_root_dir=root_path, split='test', group=[1, 0]),
                 WB_DomainTest(dataset_root_dir=root_path, split='test', group=[1, 1]),
                 ]

    test_rand_data = WB_RandomData(dataset_root_dir=root_path,
                                    all_split=None, sample_num=args.samples)


    print('datapath', root_path)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    def get_loader_list(data_list):
        ans=[]
        for each in data_list:
            ans.append(torch.utils.data.DataLoader(each, batch_size=args.eval_batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=train_sampler))
        return ans

    val_loader_list = get_loader_list(val_data_list)
    test_loader_list = get_loader_list(test_data_list)

    random_loader = torch.utils.data.DataLoader(
        test_rand_data, batch_size=args.eval_batch_size*args.samples, shuffle=True,
        num_workers=args.workers*2, pin_memory=True, sampler=train_sampler)

    from models.resnet import resnet50, FDC5
    resnet = resnet50()
    checkpoint = torch.load('./resnet50.pth')
    resnet.load_state_dict(checkpoint)

    resnet = nn.DataParallel(resnet).cuda()

    latent_dim=2048
    print('fdc5')
    classifier = FDC5(hidden_dim=latent_dim, cat_num=outdim, drop_xp=args.drop_xp, drop_xp_ratio=args.drop_xp_ratio, middle_hidden=args.middle_hidden).cuda()
    classifier = nn.DataParallel(classifier)
    classifier.train()

    params = list(classifier.parameters()) + list(resnet.parameters()) 
    opt = torch.optim.Adam(params, lr=args.lr_max)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.CrossEntropyLoss().cuda()

    best_test_robust_acc = 0
    best_val_robust_acc = 0

    start_epoch = 1

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    epochs = 10
    print(epochs, 'FD5')

    ssl_epoch = args.ssl_epoch

    beta=1

    run_train = True
    if run_train:
        train_acc_list=[]
        testfd_acc_list=[]
        testvani_acc_list=[]

        start_train_bk = 0 # disable this
        for epoch in range(start_epoch, epochs+1):

            resnet.train()
            classifier.train()

            start_time = time.time()
            train_loss = 0
            train_contrastive_loss = 0
            train_vae_loss = 0
            train_acc = 0
            train_aux_acc = 0

            torch.cuda.empty_cache()

            train_n = 0
            for i, batch in enumerate(train_loader):
                if i==100 and args.shorter:
                    break
                if args.eval:
                    break
                x1,  Xp, y = batch
                x1 = x1.cuda()

                Xp = Xp.cuda()
                y = y.cuda()

                aux_prediction, feature = resnet(x1)
                p=0.5
                binomial = torch.distributions.binomial.Binomial(probs=1 - p)
                feature = feature * binomial.sample(feature.size()).cuda() * (1.0 / (1 - p))

                # flag=True # detach always
                prediction = classifier(feature, Xp, False, random_detach=args.detach)

                # aux_prediction = aux_classifier(latent)
                aux_loss = criterion2(aux_prediction, y)

                cl_loss = criterion(prediction, y)

                loss = cl_loss #+ aux_loss

                train_loss += loss.item() * y.size(0)
                train_acc += (prediction.max(1)[1] == y).sum().item()
                train_aux_acc += (aux_prediction.max(1)[1] == y).sum().item()


                train_n += y.size(0)
                opt.zero_grad()
                loss.backward()
                opt.step()

            print('train loss', train_loss/train_n)
            print('start eval')
            train_time = time.time()
            torch.cuda.empty_cache()

            classifier.eval()
            resnet.eval()

            val_result=[]
            for each_val_loader in val_loader_list:
                test_robust_loss = 0
                test_robust_acc = 0
                test_n = 0
                with torch.no_grad():
                    for i, bs_pair in enumerate(zip(each_val_loader, random_loader)):
                        batch, batch_rand = bs_pair
                        X, y = batch
                        Xp, _ = batch_rand

                        X = X.cuda()
                        y = y.cuda()
                        Xp = Xp.cuda()

                        aux_prediction, feature = resnet(X)


                        # TODO: x_pair
                        bs_m = feature.size(0)
                        j=0
                        logit_compose = classifier(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :])

                        for jj in range(args.samples-1):
                            logit_compose = logit_compose + classifier(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :])  # TODO:

                        test_robust_acc += (logit_compose.max(1)[1] == y).sum().item()
                        test_robust_loss += loss.item() * y.size(0)
                        test_n += y.size(0)

                val_result.append(test_robust_acc/test_n)
            avg = 0.7295099061522419 * val_result[0] + 0.0383733055265902 * val_result[1] + 0.01167883211678832*val_result[2] + 0.22043795620437956*val_result[3]
            worst = min(val_result)

            train_time = time.time()

            classifier.eval()
            print(avg, worst)

            vani_avg=0
            vani_worst=0

            test_time = time.time()

            print("epoch", epoch, " train acc", train_acc / train_n,
                  "test vanilla", vani_avg, vani_worst,
                  "test fd", avg, worst)
            torch.cuda.empty_cache()

            train_acc_list.append(train_acc / train_n)
            testfd_acc_list.append(worst)

            if (epoch + 1) % args.chkpt_iters == 0 or epoch + 1 == epochs:
                torch.save({'state_dict_classifier': classifier.state_dict(), 'state_dict_resnet': resnet.state_dict(),},
                           os.path.join(args.fname, f'model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))

                # save best
            if worst > best_test_robust_acc:
                torch.save({
                    'state_dict_classifier': classifier.state_dict(),
                    'state_dict_resnet': resnet.state_dict(),
                    'test_robust_acc': test_robust_acc / test_n,
                    'test_robust_loss': test_robust_loss / test_n,
                    # 'test_loss': test_vanilla_loss / test_n,
                    # 'test_acc': test_vanilla_acc / test_n,
                }, os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = worst
                # paired_vanilla_acc = test_vanilla_acc / test_n_v
            torch.cuda.empty_cache()

        print(train_acc_list)
        print(testfd_acc_list)
        print(testvani_acc_list)

        print("best robust acc", best_test_robust_acc,  "\n\n\n") #"corresponding vanilla", paired_vanilla_acc,
        checkpoint = torch.load(os.path.join(args.fname, f'model_best.pth'))

    else:
        checkpoint = torch.load('waterbird_model_checkpoint_78.pth')  # one random run
        
    # checkpoint = torch.load('./resnet50.pth')
    resnet.load_state_dict(checkpoint['state_dict_resnet'])
    classifier.load_state_dict(checkpoint['state_dict_classifier'])

    classifier.eval()
    resnet.eval()

    print("fd samples", args.samples)

    add_n=True
    p = 0.9

    avg=0
    worst=0


    print(f'p={p}')
    # If want to change Xp batchsize, simply change X batchsize, as Xp size is set to be the same as X in `bs_m = feature.size(0)`

    ######################
    # Causal inference
    val_result = []
    for each_val_loader in test_loader_list:
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        with torch.no_grad():
            for i, bs_pair in enumerate(zip(each_val_loader, random_loader)):
                batch, batch_rand = bs_pair
                X, y = batch
                Xp, _ = batch_rand

                X = X.cuda()
                y = y.cuda()
                Xp = Xp.cuda()

                aux_prediction, feature = resnet(X)

                # TODO: x_pair
                bs_m = feature.size(0)
                j = 0

                if add_n:
                    binomial = torch.distributions.binomial.Binomial(probs=1 - p)
                    fea = feature * binomial.sample(feature.size()).cuda() * (1.0 / (1 - p))
                else:
                    fea = feature
                logit_compose = classifier(fea, Xp[j * bs_m:(j + 1) * bs_m, :, :, :])

                for jj in range(args.samples - 1):
                    if add_n:
                        binomial = torch.distributions.binomial.Binomial(probs=1 - p)
                        fea = feature * binomial.sample(feature.size()).cuda() * (1.0 / (1 - p))
                    else:
                        fea = feature
                    logit_compose = logit_compose + classifier(fea, Xp[j * bs_m:(j + 1) * bs_m, :, :, :])  # TODO:

                test_robust_acc += (logit_compose.max(1)[1] == y).sum().item()
                # test_robust_loss += loss.item() * y.size(0)
                test_n += y.size(0)

        val_result.append(test_robust_acc / test_n)
    avg = 0.7295099061522419 * val_result[0] + 0.0383733055265902 * val_result[1] + 0.01167883211678832 * val_result[
        2] + 0.22043795620437956 * val_result[3]
    worst = min(val_result)

    print('Test results on Four Groups:', val_result)

    print("test our method Average Accuracy:", avg, "Worst Group Accuracy:", worst)
    ######################

    train_time = time.time()

    classifier.eval()

    ##########################
    # Baseline inference without Causal inference
    val_vanilla_result = []
    for each_val_loader in test_loader_list:
        test_vanilla_acc = 0
        test_vanilla_aux_acc = 0
        test_vanilla_loss = 0
        test_n_v = 0
        with torch.no_grad():
            for i, batch in enumerate(each_val_loader):
                X, y = batch

                X = X.cuda()
                y = y.cuda()

                aux_prediction, feature = resnet(X)

                # TODO: x_pair
                bs_m = feature.size(0)
                j = 0
                logit_compose = classifier(z1=feature, z2=X, test=False)

                test_vanilla_aux_acc += (aux_prediction.max(1)[1] == y).sum().item()
                test_vanilla_acc += (logit_compose.max(1)[1] == y).sum().item()
                test_n_v += y.size(0)

        val_vanilla_result.append(test_vanilla_acc / test_n_v)
    vani_avg = 0.7295099061522419 * val_vanilla_result[0] + 0.0383733055265902 * val_vanilla_result[
        1] + 0.01167883211678832 * \
               val_vanilla_result[2] + 0.22043795620437956 * val_vanilla_result[3]
    vani_worst = min(val_vanilla_result)
    print(
          "test baseline Accuracy:", vani_avg, "Worst Group Accuracy:", vani_worst, 
          "test Our Causal Accuracy:", avg, "Worst Group Accuracy:", worst)
    print('\n\n\n')
    ##########################

    def ensemble_baseline(ensemble_num):
        val_vanilla_result = []
        for each_val_loader in test_loader_list:
            test_vanilla_acc = 0
            test_vanilla_aux_acc = 0
            test_vanilla_loss = 0
            test_n_v = 0
            with torch.no_grad():
                for i, batch in enumerate(each_val_loader):
                    X, y = batch

                    X = X.cuda()
                    y = y.cuda()

                    aux_prediction, feature = resnet(X)

                    for ensemble_cnt in range(ensemble_num):
                        binomial = torch.distributions.binomial.Binomial(probs=1 - p)
                        fea = feature * binomial.sample(feature.size()).cuda() * (1.0 / (1 - p))

                        # TODO: x_pair
                        bs_m = fea.size(0)
                        j = 0
                        logit_compose = classifier(z1=fea, z2=X, test=False)

                        if ensemble_cnt==0:
                            logit_sum = logit_compose
                        else:
                            logit_sum = logit_sum + logit_compose

                    # aux_prediction = aux_classifier(latent)

                    test_vanilla_aux_acc += (aux_prediction.max(1)[1] == y).sum().item()
                    test_vanilla_acc += (logit_sum.max(1)[1] == y).sum().item()
                    test_n_v += y.size(0)

            val_vanilla_result.append(test_vanilla_acc / test_n_v)
        vani_avg = 0.7295099061522419 * val_vanilla_result[0] + 0.0383733055265902 * val_vanilla_result[
            1] + 0.01167883211678832 * \
                val_vanilla_result[2] + 0.22043795620437956 * val_vanilla_result[3]
        vani_worst = min(val_vanilla_result)
        print(" train acc ensemble", ensemble_num, 
            "\ntest baseline average Accuracy:", vani_avg, "Worst Group Accuracy:", vani_worst)

    # ensemble_baseline(1)
    # ensemble_baseline(5)
    # ensemble_baseline(10)
    # ensemble_baseline(25)
    # ensemble_baseline(50)
    # ensemble_baseline(100)
    # ensemble_baseline(200)
    # ensemble_baseline(500)
    # ensemble_baseline(1000)




if __name__ == "__main__":
    main()