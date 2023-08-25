import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
#import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader

import lib.dist as dist
import lib.utils as utils
from lib.datasets import return_data
from lib.flows import FactorialNormalizingFlow

from elbo_decomposition import elbo_decomposition
#from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401
from torchvision.utils import save_image


class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        #ch_list = [64,128,128,256,256]
        ch_list = [32,64,64,128,128]
        #self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.conv1 = nn.Conv2d(1, ch_list[0], 1, 1, 6)     #64*128 *128
        self.bn1 = nn.BatchNorm2d(ch_list[0])
        self.conv2 = nn.Conv2d(ch_list[0], ch_list[1], 3, 2, 1)  #128 * 64 * 64
        self.bn2 = nn.BatchNorm2d(ch_list[1])
        self.conv3 = nn.Conv2d(ch_list[1], ch_list[2], 3, 2, 1)  #128 * 32 * 32
        self.bn3 = nn.BatchNorm2d(ch_list[2])
        self.conv4 = nn.Conv2d(ch_list[2], ch_list[3], 3, 2, 1)  #256 * 16 * 16
        self.bn4 = nn.BatchNorm2d(ch_list[3])
        self.conv5 = nn.Conv2d(ch_list[3], ch_list[4], 3, 2, 1)  #258 * 8 * 8
        self.bn5 = nn.BatchNorm2d(ch_list[4])
        self.conv_z = nn.Conv2d(ch_list[4], output_dim, 8)   # 20 * 1 * 1

        # setup the non-linearity    
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x #.view(-1, 1, 64, 64)                    #1* 116 * 116
        h = self.act(self.bn1(self.conv1(h)))          #64*128 *128
        h = self.act(self.bn2(self.conv2(h)))          #128 * 64 * 64
        h = self.act(self.bn3(self.conv3(h)))          #128 * 32 * 32
        h = self.act(self.bn4(self.conv4(h)))          #256 * 16 * 16
        h = self.act(self.bn5(self.conv5(h)))          #256 * 8 * 8
        z = self.conv_z(h).view(x.size(0), self.output_dim)     # 20 
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        #ch_list = [64,128,128,256,256]
        ch_list = [32,64,64,128,128]
        self.conv1 = nn.ConvTranspose2d(input_dim, ch_list[-1], 8, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(ch_list[-1])
        self.conv2 = nn.ConvTranspose2d(ch_list[-1], ch_list[-2], 4, 2, 1)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(ch_list[-2])
        self.conv3 = nn.ConvTranspose2d(ch_list[-2], ch_list[-3], 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(ch_list[-3])
        self.conv4 = nn.ConvTranspose2d(ch_list[-3], ch_list[-4], 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(ch_list[-4])
        self.conv5 = nn.ConvTranspose2d(ch_list[-4], ch_list[-5], 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(ch_list[-5])
        #self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)    
        self.conv_final = nn.ConvTranspose2d(ch_list[-5], 1, 1, 1, 6)     #1-> 3

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)       #10 * 1* 1
        h = self.act(self.bn1(self.conv1(h)))        #256 * 8*8
        h = self.act(self.bn2(self.conv2(h)))        #256 * 16 * 16
        h = self.act(self.bn3(self.conv3(h)))        #128 * 32 *32
        h = self.act(self.bn4(self.conv4(h)))        #128 * 64 * 64
        h = self.act(self.bn5(self.conv5(h)))        #64 * 128 * 128
        mu_img = self.conv_final(h)                  #1 * 116 * 116
        return mu_img


class VAE(nn.Module):
    def __init__(self, args, z_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, conv=False, mss=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb =args.lamb      #1-lambda = gamma
        self.beta = args.beta   #1
        self.alpha =args.alpha
        self.mss = mss
        self.x_dist = dist.Bernoulli()
        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        #x = x.view(x.size(0), 1, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)      #contain mean and logvar
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)        
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 1, 116, 116)     
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        #x = x.view(batch_size, 1, 64, 64)     #??
        prior_params = self._get_prior_params(batch_size)                  #batch-size * dim_z * 2     
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        #if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
        #    return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))      #formular 3
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:     #into
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))               #formular 3
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:            
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    self.alpha * (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)            ##term 3      self.beta * (logqz - logqz_prodmarginals) is term 2     (logqz_condx - logqz) is term 1 for MI
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, args):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    #sample_mu = model.model_sample(batch_size=100).sigmoid()
    #sample_mu = sample_mu
    #images = list(sample_mu.view(-1, 3, 64, 64).data.cpu())
    #win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat([
        test_imgs, reco_imgs], 0)
        #win_test_reco = vis.images(
        #list(test_reco_imgs.contiguous().view(-1, 3, 64, 64).data.cpu()), 10, 2,
        #opts={'caption': 'test reconstruction image'}, win=win_test_reco)
    save_image(test_reco_imgs, f'{args.model_name}_rec.png', nrow=10, pad_value =1, normalize = False)    

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[:4]
    batch_size, z_dim = zs.size()
    xs = []
    steps = 13
    delta = torch.autograd.Variable(torch.linspace(-6, 6, steps), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(steps, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = torch.cat(xs, 0).data.cpu()
    #win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)
    save_image(xs, f'{args.model_name}_latent.png', nrow=steps, pad_value =1, normalize = False)

def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='brain', type=str, help='dataset name',
        choices=['brain','celeba', '3dchairs'])
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=10, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')    #
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_true')      #
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='test1')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    #added
    parser.add_argument('--dset_dir', default='con_mat_all.npy', type=str, help='dataset directory')
    parser.add_argument('--image_size', default=116, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')


    parser.add_argument('--model_dir', default="", type=str )     
    parser.add_argument('--model_name', default="", type=str)   
    parser.add_argument('--alpha', default=1, type=float)    
    parser.add_argument('--lamb', default=0, type=float)   
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"
    #torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    #add

    # data loader
    train_loader = return_data(args)    #modified

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    vae = VAE(args, z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss).to('cpu')
    
    if args.model_dir:
        ckpt_dict = torch.load(args.model_dir, map_location='cpu')
        vae.load_state_dict(ckpt_dict['state_dict'])
        print('load weights')

    vae = vae.to(device)
    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env=args.save, port=4500)

    train_elbo = []

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, x in enumerate(train_loader):
            iteration += 1
            batch_time = time.time()
            vae.train()
            #anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU
            x =x.to(device) #x.cuda(async=True)
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            # do ELBO gradient and accumulate loss
            obj, elbo = vae.elbo(x, dataset_size)
            #if utils.isnan(obj).any():
            #    raise ValueError('NaN spotted in objective.')
            obj = obj.mean()
            obj.mul(-1).backward()
            elbo_running_mean.update(elbo.mean().data)    #modified (elbo.mean().data[0])
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print('[iteration %03d] time: %.2f dim %.2f alpha %.2f beta %.2f gamma %.2f  loss %.2f recon_loss %.2f ELBO: current %.4f ' % (
                    iteration, time.time() - batch_time, 
                    args.latent_dim, vae.alpha, vae.beta, (1-vae.lamb), 
                    -obj, abs(obj-elbo_running_mean.val), elbo_running_mean.val))
                vae.eval()

                # plot training and test ELBOs
                if args.visdom:
                    display_samples(vae, x, vis)
                    plot_elbo(train_elbo, vis)
            
                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, args.save, iteration)
                #eval('plot_vs_gt_' + args.dataset)(vae, train_loader.dataset,
                #    os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)))

    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, iteration)
    dataset_loader = DataLoader(train_loader.dataset, batch_size=16, num_workers=1, shuffle=False)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
    #eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'))
    return vae

def get_latent_code():
    #get the latent code for the first 10 image for exploring the latent code meaning

    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='celeba', type=str, help='dataset name',
        choices=['celeba', '3dchairs'])
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=10, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')    #
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_true')      #
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='test1')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    #added
    parser.add_argument('--dset_dir', default='/storage/changyu/datasets', type=str, help='dataset directory')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')


    parser.add_argument('--model_dir', default="", type=str )     
    parser.add_argument('--model_name', default="", type=str)    
    parser.add_argument('--alpha', default=1, type=float)    
    parser.add_argument('--lamb', default=0, type=float)   
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
        
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"
    #torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    #add

    # data loader
    train_loader = return_data(args)    #modified

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    vae = VAE(args, z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss).to('cpu')
    ckpt_dict = torch.load(args.model_dir, map_location='cpu')
    vae.load_state_dict(ckpt_dict['state_dict'])
    vae = vae.to(device)
    vae.eval()
    x = next(iter(train_loader)).to(device)
    #zs, _ = vae.encode(x)
    #return zs[:10]
    #vis = visdom.Visdom()
    display_samples(vae, x, args)

if __name__ == '__main__':
    model = main()
    #get_latent_code()
