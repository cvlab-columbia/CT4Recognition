import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import random

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=False)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

log_interval = 50

torch.cuda.empty_cache()


def make_environment(imgs, labels, e, train=True):
    # labels = labels.float()
    # imgs = imgs.reshape((-1, 28, 28))[:, ::2, ::2]
    # imgs2 = torch.stack([imgs, imgs, imgs], dim=1)
    # imgs = torch.stack([imgs, imgs, imgs], dim=1)

    trainloader = DataLoader(
        datasets.MNIST(
            "../dataset",
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
        ),
        batch_size=128,
        shuffle=True,
    )

    imgs_list = []
    imgs_label_list = []
    for batch_idx, (imgs, labels) in enumerate((trainloader)):
        imgs_list.append(imgs)
        imgs_label_list.append(labels)
    imgs = torch.cat(imgs_list, dim=0)
    imgs2 = torch.cat(imgs_list, dim=0)
    labels = torch.cat(imgs_label_list, dim=0)
    print(imgs.size(), 'img size', labels.size())

    imgs[labels == 0, 0, :, :] = torch.ones_like(imgs[labels == 0, 0, :, :])

    imgs[labels == 0, 1, :, :] = torch.ones_like(imgs[labels == 0, 1, :, :])

    imgs[labels == 1, 1, :, :] = torch.ones_like(imgs[labels == 1, 1, :, :])
    imgs[labels == 1, 2, :, :] = torch.ones_like(imgs[labels == 1, 2, :, :])

    imgs[labels == 2, 2, :, :] = torch.ones_like(imgs[labels == 2, 2, :, :])
    imgs[labels == 2, 0, :, :] = torch.ones_like(imgs[labels == 2, 0, :, :])

    imgs[labels == 3, 0, :, :] = torch.ones_like(imgs[labels == 3, 0, :, :])

    imgs[labels == 4, 1, :, :] = torch.ones_like(imgs[labels == 4, 1, :, :])

    imgs[labels == 5, 2, :, :] = torch.ones_like(imgs[labels == 5, 2, :, :])

    imgs[labels == 6, 0, :, :] = torch.zeros_like(imgs[labels == 6, 0, :, :])
    imgs[labels == 6, 1, :, :] = torch.zeros_like(imgs[labels == 6, 1, :, :])

    imgs[labels == 7, 1, :, :] = torch.zeros_like(imgs[labels == 7, 1, :, :])
    imgs[labels == 7, 2, :, :] = torch.zeros_like(imgs[labels == 7, 2, :, :])

    imgs[labels == 8, 2, :, :] = torch.zeros_like(imgs[labels == 8, 2, :, :])
    imgs[labels == 8, 0, :, :] = torch.zeros_like(imgs[labels == 8, 0, :, :])

    imgs[labels == 9, 1, :, :] = torch.zeros_like(imgs[labels == 9, 1, :, :])

    ######
    imgs2[labels == 5, 0, :, :] = torch.ones_like(imgs2[labels == 5, 0, :, :])
    imgs2[labels == 5, 1, :, :] = torch.zeros_like(imgs2[labels == 5, 1, :, :])

    imgs2[labels == 3, 1, :, :] = torch.ones_like(imgs2[labels == 3, 1, :, :])
    imgs2[labels == 3, 2, :, :] = torch.zeros_like(imgs2[labels == 3, 2, :, :])

    imgs2[labels == 4, 2, :, :] = torch.ones_like(imgs2[labels == 4, 2, :, :])
    imgs2[labels == 4, 0, :, :] = torch.zeros_like(imgs2[labels == 4, 0, :, :])

    imgs2[labels == 1, 0, :, :] = torch.zeros_like(imgs2[labels == 1, 0, :, :])

    imgs2[labels == 2, 1, :, :] = torch.zeros_like(imgs2[labels == 2, 1, :, :])

    imgs2[labels == 0, 2, :, :] = torch.zeros_like(imgs2[labels == 0, 2, :, :])

    imgs2[labels == 6, 0, :, :] = torch.zeros_like(imgs2[labels == 6, 0, :, :])
    imgs2[labels == 6, 1, :, :] = torch.ones_like(imgs2[labels == 6, 1, :, :])

    imgs2[labels == 7, 1, :, :] = torch.zeros_like(imgs2[labels == 7, 1, :, :])
    imgs2[labels == 7, 2, :, :] = torch.ones_like(imgs2[labels == 7, 2, :, :])

    imgs2[labels == 8, 2, :, :] = torch.zeros_like(imgs2[labels == 8, 2, :, :])
    imgs2[labels == 8, 0, :, :] = torch.ones_like(imgs2[labels == 8, 0, :, :])

    imgs2[labels == 9, 1, :, :] = torch.ones_like(imgs2[labels == 9, 1, :, :])
    #######

    print(imgs.size(), imgs2.size())
    imgs = torch.cat([imgs, imgs2], dim=0)

    print(labels.size())
    labels = torch.cat([labels, labels], dim=0)

    img0 = imgs[labels == 0]
    img1 = imgs[labels == 1]
    img2 = imgs[labels == 2]
    img3 = imgs[labels == 3]
    img4 = imgs[labels == 4]
    img5 = imgs[labels == 5]
    img6 = imgs[labels == 6]
    img7 = imgs[labels == 7]
    img8 = imgs[labels == 8]
    img9 = imgs[labels == 9]

    pair = {0: img0, 1: img1, 2: img2, 3: img3, 4: img4, 5: img5, 6: img6, 7: img7, 8: img8, 9: img9}
    for k in pair.keys():
        pair[k] = (pair.get(k).float() / 1.).cuda()

    return (imgs.float() / 1.).cuda(), labels[:, None].cuda(), pair


def test_env(test_imgs, labels):
    # labels = labels.float()
    # test_imgs = test_imgs.reshape((-1, 28, 28))[:, ::2, ::2]
    # test_imgs = torch.stack([test_imgs, test_imgs, test_imgs], dim=1)

    testloader = DataLoader(
        datasets.MNIST(
            "../dataset",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
        ),
        batch_size=128,
        shuffle=True,
    )
    imgs_list = []
    imgs_label_list = []
    for batch_idx, (imgs, labels) in enumerate((testloader)):
        imgs_list.append(imgs)
        imgs_label_list.append(labels)
    test_imgs = torch.cat(imgs_list, dim=0)
    # imgs2 = torch.cat(imgs_list, dim=0)
    labels = torch.cat(imgs_label_list, dim=0)
    print(imgs.size(), 'img size', labels.size())

    total = test_imgs.size(0)
    ee = 20
    for cnt in range(total // ee):
        channel = np.random.choice(3, 2)
        color = np.random.sample([3]) > 0.5
        for ch in channel:
            if color[ch]:
                test_imgs[cnt * ee:(cnt + 1) * ee, ch, :, :] = torch.ones_like(
                    test_imgs[cnt * ee:(cnt + 1) * ee, ch, :, :])
            else:
                test_imgs[cnt * ee:(cnt + 1) * ee, ch, :, :] = torch.zeros_like(
                    test_imgs[cnt * ee:(cnt + 1) * ee, ch, :, :])
    return (test_imgs.float() / 1.).cuda(), labels[:, None].cuda()


def toCPU(x):
    return x.detach().cpu().numpy()


train_set, train_label, pair = make_environment(mnist_train[0], mnist_train[1], 0.2)
i3, l3 = test_env(mnist_val[0], mnist_val[1])


test_c, test_l_c, _ = make_environment(mnist_val[0], mnist_val[1], 0, train=False)

in_dim = 14 ** 2 * 3
in_dim = 32 ** 2 * 3
emb_dim = 5
hidden_dim = 1024 * 2 
z_hidden = 32  
stride = 8
sub_in_dim = 4 ** 2 * 3
# sub_in_dim = 3 ** 2 * 3
beta = 1
bs = 512
bs_t = 1000

device = torch.device("cuda")


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_hidden)
        self.fc22 = nn.Linear(hidden_dim, z_hidden)
        self.fc3 = nn.Linear(z_hidden, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, in_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, in_dim))
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


z_middle_fc = 256


class FDC(nn.Module):
    def __init__(self):
        super(FDC, self).__init__()

        self.fc_c1 = nn.Linear(z_hidden + sub_in_dim, z_middle_fc)
        self.fc_c2 = nn.Linear(z_middle_fc, 10)

    def forward(self, z1, z2, test=True):

        if test:
            bs = z1.size(0)
            bs2 = z2.size(0)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs2, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs * bs2, -1)

            h3 = F.relu(self.fc_c1(hh))
            out = self.fc_c2(h3)
            out = out.view(bs, bs2, -1)

            return torch.sum(out, dim=1)
        else:
            h = torch.cat((z1, z2), dim=1)
            h3 = F.relu(self.fc_c1(h))
            out = self.fc_c2(h3)
            return out


model = VAE().to(device)
classifier = FDC().to(device)
optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=3e-4)

xent_criteria = nn.CrossEntropyLoss()


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, in_dim), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD * beta


def train(model, classifier, epoch):
    model.train()
    classifier.train()
    train_loss = 0
    train_num = train_set.size(0)
    total_bs = train_num // bs
    # print('t bs', total_bs)
    correct = 0
    cnt = 0
    import random

    total = [l.size(0) for k, l in pair.items()]

    ind = [_ for _ in range(train_num)]
    random.shuffle(ind)
    train_set_tmp = train_set[ind]
    train_label_tmp = train_label[ind]

    for i in range(total_bs):
        data = train_set_tmp[i * bs:(i + 1) * bs]
        target = train_label_tmp[i * bs:(i + 1) * bs].long()

        data_bd = torch.zeros_like(data)
        for each in range(data.size(0)):
            lt = target[each].item()
            num_t = total[int(lt)]
            c = random.randrange(0, num_t, 1)
            data_bd[each] = pair[int(lt)][c]

        data = data.to(device)
        target = target.to(device)
        target = target[:, 0]
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fdvar = mu + eps * std

        data_sub = data_bd[:, :, ::stride, ::stride]

        fdvar = fdvar.detach() # stop gradient from classifier to VAE feature extractor

        logit_compose = classifier(fdvar, data_sub.reshape(bs, -1), False)
        # when training, in p(y|x,z), the x and z needs to be consistent, they must be produced by the same category.

        loss = loss_function(recon_batch, data, mu, logvar)
        xent_loss = xent_criteria(logit_compose, target)
        if epoch < 30:
            # pretrain VAE first
            loss = loss
        else:
            # then train classify on the pretrained VAE representation
            loss = loss + xent_loss 
        loss.backward()
        train_loss += loss.item()

        optimizer.step()

        _, predicted = torch.max(logit_compose, 1)
        correct += (predicted == target).sum().item()
        cnt += target.size(0)

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.6f}'.format(
                epoch, i * len(data), total_bs,
                       bs * 100. * i / train_num,
                       loss.item() / len(data), correct * 100.0 / cnt))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / train_num))


import random


def test(model, classifier, epoch, test_i, test_l, train_data, fd_use_train=False, fd_use_test=True, xp_num=None, Loop_num=10, noise_level=1):
    logits_T = []
    model.eval()
    classifier.eval()
    test_loss = 0
    test_num = test_i.size(0)
    bs_num = test_num // bs_t
    if bs_num * bs_t != test_num:
        bs_num += 1

    train_num = train_data.size(0)

    cnt = 0
    correct = 0
    total_test = test_i.size(0)
    with torch.no_grad():
        for i in range(bs_num):
            data = test_i[i * bs_t:(i + 1) * bs_t]
            target = test_l[i * bs_t:(i + 1) * bs_t].long()
            data = data.to(device)
            target = target.to(device)
            target = target[:, 0]
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * noise_level
            fdvar = mu + eps * std

            weight = torch.exp(-torch.mean(eps ** 2))

            if fd_use_test and fd_use_train:
                data_sub = data[:, :, ::stride, ::stride]
                for each in range(data_sub.size(0) // 2):
                    c = random.randrange(0, train_num, 1)
                    data_sub[each] = train_data[c, :, ::stride, ::stride]
            elif fd_use_test:
                data_sub = data[:, :, ::stride, ::stride]
            elif fd_use_train:
                if xp_num is not None:
                    tmp = data[:, :, ::stride, ::stride]
                    data_sub = torch.zeros((xp_num, tmp.size(1), tmp.size(2), tmp.size(3))).cuda()
                else:
                    data_sub = torch.zeros_like(data[:, :, ::stride, ::stride])
                for each in range(data_sub.size(0)):
                    c = random.randrange(0, train_num, 1)
                    data_sub[each] = train_data[c, :, ::stride, ::stride]

            print('xp bs', data_sub.size(0))
            logit_compose = weight * classifier(fdvar, data_sub.reshape(data_sub.size(0), -1))  # TODO:

            for jj in range(Loop_num):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)

                weight = torch.exp(-torch.mean(eps ** 2))
                fdvar = mu + eps * std

                if fd_use_test and fd_use_train:
                    data_sub = data[:, :, ::stride, ::stride]
                    for each in range(data_sub.size(0) // 2):
                        c = random.randrange(0, train_num, 1)
                        data_sub[each] = train_data[c, :, ::stride, ::stride]
                    for each in range(data_sub.size(0) // 2, data_sub.size(0)):
                        c = random.randrange(0, total_test, 1)
                        data_sub[each] = test_i[c, :, ::stride, ::stride]
                elif fd_use_test:
                    data_sub = data[:, :, ::stride, ::stride]
                    for each in range(data_sub.size(0)):
                        c = random.randrange(0, total_test, 1)
                        data_sub[each] = test_i[c, :, ::stride, ::stride]
                elif fd_use_train:
                    if xp_num is not None:
                        tmp = data[:, :, ::stride, ::stride]
                        data_sub = torch.zeros((xp_num, tmp.size(1), tmp.size(2), tmp.size(3))).cuda()
                    else:
                        data_sub = torch.zeros_like(data[:, :, ::stride, ::stride])
                    for each in range(data_sub.size(0)):
                        c = random.randrange(0, train_num, 1)
                        data_sub[each] = train_data[c, :, ::stride, ::stride]

                logit_compose = logit_compose + weight * classifier(fdvar, data_sub.reshape(data_sub.size(0), -1))

            logits_T.append(logit_compose)

            _, predicted = torch.max(logit_compose, 1)
            correct += (predicted == target).sum().item()
            cnt += target.size(0)

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(bs_t, 3, 32, 32)[:n]])
                comparison = torch.clamp(comparison, 0, 1)
                save_image(comparison.cpu(),
                           './' + str(epoch) + '.png', nrow=n)

    test_loss /= test_num
    print('====> Test set loss: {:.4f}  Our Causal Acc: {:.4f}'.format(test_loss, correct * 100.0 / cnt), 'use train',
          fd_use_train, 'use test', fd_use_test)
    return logits_T, correct * 100.0 / cnt





def test_bl(model, classifier, epoch, test_i, test_l):
    model.eval()
    classifier.eval()
    test_loss = 0
    test_num = test_i.size(0)
    bs_num = test_num // bs_t
    if bs_num * bs_t != test_num:
        bs_num += 1

    cnt=0
    correct=0
    total_test = test_i.size(0)
    with torch.no_grad():
        for i in range(bs_num):
            data = test_i[i * bs_t:(i + 1) * bs_t]
            target = test_l[i * bs_t:(i + 1) * bs_t].long()
            data = data.to(device)
            target = target.to(device)
            target = target[:, 0]
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            fdvar = mu

            data_sub = data[:, :, ::stride, ::stride]
            logit_compose = classifier(fdvar, data_sub.reshape(data_sub.size(0), -1), test=False)  #TODO:


            _, predicted = torch.max(logit_compose, 1)
            correct += (predicted == target).sum().item()
            cnt += target.size(0)

    test_loss /= test_num
    print('====> Vanilla Test set loss: {:.4f}  Acc: {:.4f}'.format(test_loss, correct*100.0/cnt))



def test_bl_ensemble(model, classifier, epoch, test_i, test_l, ensemble_times=100):
    model.eval()
    classifier.eval()
    test_loss = 0
    test_num = test_i.size(0)
    bs_num = test_num // bs_t
    if bs_num * bs_t != test_num:
        bs_num += 1

    cnt=0
    correct=0
    total_test = test_i.size(0)
    with torch.no_grad():
        for i in range(bs_num):
            data = test_i[i * bs_t:(i + 1) * bs_t]
            target = test_l[i * bs_t:(i + 1) * bs_t].long()
            data = data.to(device)
            target = target.to(device)
            target = target[:, 0]
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            for ensemble_num in range(ensemble_times):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                fdvar = mu + eps * std

                r = random.randrange(0, stride)
                data_sub = data[:, :, r::stride, r::stride]
                logit_compose = classifier(fdvar, data_sub.reshape(data_sub.size(0), -1), test=False)  #TODO:

                if ensemble_num==0:
                    sum_logit = logit_compose
                else:
                    sum_logit = logit_compose + sum_logit
                
            _, predicted = torch.max(sum_logit, 1)
            correct += (predicted == target).sum().item()
            cnt += target.size(0)

    test_loss /= test_num
    print('====> Ensembled Vanilla Test set loss: {:.4f}  Acc: {:.4f}'.format(test_loss, correct*100.0/cnt))



from scipy.stats import norm
import numpy as np

vis_width = 77
epochs = 31
use_train_D = False
use_test_D = True
best_test_acc = -1
fname = 'cvpr_cmnist_s{}_z{}_fdc_l_{}'.format(stride, z_hidden, z_middle_fc)
import os


eval_only=False
if not eval_only:
    for epoch in range(1, epochs + 1):
        print("")
        train(model, classifier, epoch)
        _, acc = test(model, classifier, epoch, i3, l3, train_set, fd_use_train=True, fd_use_test=False)
        test_bl(model, classifier, epoch, i3, l3)
        if best_test_acc < acc:
            torch.save({
                'VAE': model.state_dict(),
                'FDClassifer': classifier.state_dict(),
                'test_acc': acc,
            }, fname + '_model_best.pth')
            best_test_acc = acc


    model = VAE().to(device)
    classifier = FDC().to(device)

    ck = torch.load(fname + '_model_best.pth')
    print('loading test', ck['test_acc'])
    model.load_state_dict(ck['VAE'])
    classifier.load_state_dict(ck['FDClassifer'])

    print("\n\n\n\n\n")
    _, acc = test(model, classifier, epoch, i3, l3, train_set, fd_use_train=True, fd_use_test=False)
    test_bl(model, classifier, epoch, i3, l3)

else:
    print('start loading')
    checkpoint = torch.load('cvpr_cmnist_s8_z32_fdc_l_256_model_best_508.pth')
    model.load_state_dict(checkpoint['VAE'])
    classifier.load_state_dict(checkpoint['FDClassifer'])

    classifier.eval()
    model.eval()

    # front-door
    _, acc = test(model, classifier, 0, i3, l3, train_set, fd_use_train=True, fd_use_test=False, xp_num=50, Loop_num=0, noise_level=1)
    test_bl(model, classifier, 0, i3, l3)

    # ensemble experiment, which are the baseline suggested by the reviewer
    test_bl_ensemble(model, classifier, 0, i3, l3, 1)
    test_bl_ensemble(model, classifier, 0, i3, l3, 10)
    test_bl_ensemble(model, classifier, 0, i3, l3, 100)
    test_bl_ensemble(model, classifier, 0, i3, l3, 1000)
    test_bl_ensemble(model, classifier, 0, i3, l3, 2500)
    test_bl_ensemble(model, classifier, 0, i3, l3, 5000)
    test_bl_ensemble(model, classifier, 0, i3, l3, 7500)
    test_bl_ensemble(model, classifier, 0, i3, l3, 10000)







