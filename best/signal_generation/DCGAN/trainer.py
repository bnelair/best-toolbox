# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import yaml

from datetime import datetime

from best.signal_generation.DCGAN.model import Discriminator, Generator, DiscriminatorTimeFreq
from best.signal_generation.DCGAN.dataset import DatasetTimeDomain
from best._config import ObjDictToDict

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from best.signal import PSD


class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.model_name = self.cfg.NAME
        exec(
        f"from {'.'.join(self.cfg.TRAIN.DATASET.split('.')[:-1])} import {self.cfg.TRAIN.DATASET.split('.')[-1]}\n" \
        f"self.DatasetTrain={self.cfg.TRAIN.DATASET.split('.')[-1]}({self.cfg.TRAIN.PATH_DATA}, categories_keep={self.cfg.TRAIN.CATEGORY_KEEP})"
        )
        exec(
        f"from {'.'.join(self.cfg.TRAIN.DATASET.split('.')[:-1])} import {self.cfg.TRAIN.DATASET.split('.')[-1]}\n" \
        f"self.DatasetTest={self.cfg.TRAIN.DATASET.split('.')[-1]}({self.cfg.TRAIN.PATH_DATA}, categories_keep={self.cfg.TRAIN.CATEGORY_KEEP})"
        )
        self.DatasetTrain.train()
        self.DatasetTest.eval()

        self.path_report = os.path.join(self.cfg.TRAIN.PATH_REPORT, self.cfg.NAME + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.path_report_models = os.path.join(self.path_report, 'Models')
        self.path_report_images = os.path.join(self.path_report, 'Images')
        self.path_report_config = os.path.join(self.path_report, 'config.yaml')
        self.path_report_losses = os.path.join(self.path_report, 'losses.csv')
        self.path_report_losses_validation = os.path.join(self.path_report, 'losses_val.csv')

        os.mkdir(self.path_report)
        os.mkdir(self.path_report_models)
        os.mkdir(self.path_report_images)
        with open(self.path_report_config, 'w') as file:
            yaml.dump(ObjDictToDict(self.cfg), file)

        with open(self.path_report_losses, 'w') as f:
            f.write('Epoch, lr, loss_D_fake, loss_D_real, loss_G_discr, loss_G_spect\n')

        self.save_model_freq = self.cfg.TRAIN.SAVE_MODEL_EPOCH
        self.save_report_freq = self.cfg.TRAIN.SAVE_REPORT_EPOCH

        self.GPUs = self.cfg.TRAIN.GPU
        self.num_gpus = self.GPUs.__len__()
        self.minibatchsize = int(self.cfg.TRAIN.BATCH_SIZE)

        self.lr = self.cfg.TRAIN.BASE_LR

        self.epochs = self.cfg.TRAIN.EPOCHS


        self.current_epoch = 0
        self.current_iteration = 0
        self.overall_iteration = 0
        self.epoch_list = []
        self.lossKL_list = []
        self.lossMSE_list = []
        self.loss_list = []
        self.device = self.GPUs[0]

        self.Generator = Generator(n_features=self.cfg.MODEL.ARCHITECTURE.N_EMBEDDED_FEATURES, n_filters=self.cfg.MODEL.ARCHITECTURE.N_FILTERS)
        self.Discriminator = Discriminator(n_filters=self.cfg.MODEL.ARCHITECTURE.N_FILTERS)

        self.optimizerG = optim.Adam(self.Generator.parameters(), lr=self.cfg.TRAIN.LR, betas=(self.cfg.TRAIN.BETA_1, 0.999))
        self.optimizerD = optim.Adam(self.Discriminator.parameters(), lr=self.cfg.TRAIN.LR, betas=(self.cfg.TRAIN.BETA_1, 0.999))

        self.lr_schedulerG = torch.optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=self.cfg.TRAIN.DECAY_ITERATIONS, gamma=self.cfg.TRAIN.DECAY_FACTOR)
        self.lr_schedulerD = torch.optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=self.cfg.TRAIN.DECAY_ITERATIONS, gamma=self.cfg.TRAIN.DECAY_FACTOR)



        self.criterion = nn.BCELoss()
        self.criterion_MSE = nn.MSELoss()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.Generator.apply(weights_init)
        self.Discriminator.apply(weights_init)

        self.DLoaderTrain = DataLoader(self.DatasetTrain, batch_size=self.minibatchsize, shuffle=True, num_workers=self.cfg.TRAIN.CPU_COUNT_LOADERS, drop_last=False)

        self.Discriminator.cuda(self.device)
        self.Generator.cuda(self.device)

    def do_epoch(self):
        self.current_iteration = 0
        print('----------------------------------')
        print('VAE Epoch ' + str(self.current_epoch))


        self.Discriminator.train()
        self.Generator.train()

        for k, x in enumerate(self.DLoaderTrain):
            x, y, ystr = x

            x = x.float().to(self.Discriminator.dummy_param.device)
            # break


            self.Discriminator.zero_grad()
            label = torch.full((x.shape[0] * 3,), 1.0, dtype=torch.float, device=self.Discriminator.dummy_param.device)
            # label = torch.full((x.shape[0],), 1.0, dtype=torch.float, device=self.Discriminator.dummy_param.device)
            output = self.Discriminator(x.unsqueeze(1)).view(-1)
            loss_D_real = self.criterion(output, label)
            loss_D_real.backward()


            noise = torch.randn(x.shape[0], 8, 3, device=self.Discriminator.dummy_param.device)
            fake = self.Generator(noise)
            label.fill_(0.0)
            output = self.Discriminator(fake.detach()).view(-1)
            loss_D_fake = self.criterion(output, label)
            loss_D_fake.backward()
            self.optimizerD.step()


            self.Generator.zero_grad()
            label.fill_(1.0)
            output = self.Discriminator(fake).view(-1)
            loss_G1 = self.criterion(output, label)


            x_orig_fft = torch.fft.fft(x.squeeze(1)).abs()
            x_fake_fft = torch.fft.fft(fake.squeeze(1)).abs()

            x_orig_fft = torch.log10(x_orig_fft+0.0001)
            x_fake_fft = torch.log10(x_fake_fft+0.0001)

            x_fake_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0
            x_orig_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0

            x_orig_fft[torch.isnan(x_fake_fft) | torch.isinf(x_fake_fft)] = 0
            x_fake_fft[torch.isnan(x_fake_fft) | torch.isinf(x_fake_fft)] = 0

            loss_G2 = self.criterion_MSE(x_fake_fft.mean(axis=0), x_orig_fft.mean(axis=0)) * 100

            loss_G = loss_G1 + loss_G2

            loss_G.backward()
            self.optimizerG.step()

            losses = {
                'loss_D_fake': loss_D_fake.item(),
                'loss_D_real': loss_D_real.item(),
                'loss_G_discr': loss_G1.item(),
                'loss_G_spect': loss_G2.item(),
            }


            if self.overall_iteration % self.cfg.TRAIN.SAVE_REPORT_ITERATION == 0 or k==0:
                self.print_losses_to_file(losses)
                self.plot_to_file(fake, x)

            if self.overall_iteration % self.cfg.TRAIN.SAVE_MODEL_ITERATION == 0 or k==0:
                self.save_model()

            self.current_iteration += 1
            self.overall_iteration += 1

        self.current_epoch += 1
        self.lr_schedulerD.step()
        self.lr_schedulerG.step()


    def print_losses_to_file(self, losses):
        path = self.path_report_losses
        lr = self.optimizerG.param_groups[0]['lr']
        #f.write('Epoch, lr, lossG_fake, lossG_real, lossD\n')
        printStr = f'{self.current_epoch}, {lr}, ' + ', '.join([f'{i:.8f}' for k, i in losses.items()]) + '\n'
        print(self.overall_iteration, self.current_epoch, self.current_iteration, losses)
        with open(path, 'a') as f:
            f.write(printStr)


    def plot_to_file(self, fake, real):
        x_real = real[-1].detach().cpu().squeeze().numpy()
        x_fake = fake[-1].detach().cpu().squeeze().numpy()
        img_path = os.path.join(self.path_report_images, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}")
        img_path_spect = os.path.join(self.path_report_images, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_spect")

        plt.figure(figsize=(12, 4))
        ax0 = plt.subplot(2, 1, 1)
        plt.plot(x_real)
        plt.title('Real')
        plt.subplot(2, 1, 2, sharex=ax0)
        plt.plot(x_fake)
        plt.title('Fake')
        plt.savefig(img_path + '.png')
        plt.savefig(img_path + '.svg')
        plt.close()

        f, Pxx = PSD(fake.detach().cpu().numpy().squeeze(), 500)
        Pxx_fake = Pxx.mean(axis=0)

        f, Pxx = PSD(real.detach().cpu().numpy().squeeze(), 500)
        Pxx_real = Pxx.mean(axis=0)

        plt.figure(figsize=(12, 4))
        plt.semilogy(f[f<100], Pxx_real[f<100])
        plt.semilogy(f[f<100], Pxx_fake[f<100])
        plt.legend(['Real', 'Fake'])
        plt.savefig(img_path_spect + '.png')
        plt.savefig(img_path_spect + '.svg')
        plt.close()
        #plt.show()

    def save_model(self):
        PATH_D = os.path.join(self.path_report_models, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_Discriminator")
        PATH_G = os.path.join(self.path_report_models, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_Generator")
        torch.save(self.Discriminator.state_dict(), PATH_D)
        torch.save(self.Generator.state_dict(), PATH_G)

    def train(self):
        for k in range(self.epochs):
            self.do_epoch()


# class Trainer:
#     def __init__(self, config):
#         self.cfg = config
#         self.model_name = self.cfg.NAME
#         #self.DatasetTrain =
#         exec(
#         f"from {'.'.join(self.cfg.TRAIN.DATASET.split('.')[:-1])} import {self.cfg.TRAIN.DATASET.split('.')[-1]}\n" \
#         f"self.DatasetTrain={self.cfg.TRAIN.DATASET.split('.')[-1]}({self.cfg.TRAIN.PATH_DATA}, categories_keep={self.cfg.TRAIN.CATEGORY_KEEP})"
#         )
# 
#         self.path_report = os.path.join(self.cfg.TRAIN.PATH_REPORT, self.cfg.NAME + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
#         self.path_report_models = os.path.join(self.path_report, 'Models')
#         self.path_report_images = os.path.join(self.path_report, 'Images')
#         self.path_report_config = os.path.join(self.path_report, 'config.yaml')
#         self.path_report_losses = os.path.join(self.path_report, 'losses.csv')
#         self.path_report_losses_validation = os.path.join(self.path_report, 'losses_val.csv')
# 
#         os.mkdir(self.path_report)
#         os.mkdir(self.path_report_models)
#         os.mkdir(self.path_report_images)
#         with open(self.path_report_config, 'w') as file:
#             yaml.dump(ObjDictToDict(self.cfg), file)
# 
#         with open(self.path_report_losses, 'w') as f:
#             # f.write('Epoch, lr, loss_D_fake, loss_D_real, loss_DFFT_fake, loss_DFFT_real, loss_G_discr, loss_G_spect\n')
#             f.write('Epoch, lr, loss_D_fake_time, loss_D_real_freq, loss_D_fake_time, loss_D_freq, loss_G_time, loss_G_freq\n')
# 
#         self.save_model_freq = self.cfg.TRAIN.SAVE_MODEL_EPOCH
#         self.save_report_freq = self.cfg.TRAIN.SAVE_REPORT_EPOCH
# 
#         self.GPUs = self.cfg.TRAIN.GPU
#         self.num_gpus = self.GPUs.__len__()
#         self.minibatchsize = int(self.cfg.TRAIN.BATCH_SIZE)
# 
#         self.lr = self.cfg.TRAIN.BASE_LR
# 
#         self.epochs = self.cfg.TRAIN.EPOCHS
# 
# 
#         self.current_epoch = 0
#         self.current_iteration = 0
#         self.overall_iteration = 0
#         self.epoch_list = []
#         self.lossKL_list = []
#         self.lossMSE_list = []
#         self.loss_list = []
#         self.device = self.GPUs[0]
# 
#         self.Generator = Generator(n_features=self.cfg.MODEL.ARCHITECTURE.N_EMBEDDED_FEATURES, n_filters=self.cfg.MODEL.ARCHITECTURE.N_FILTERS)
#         self.Discriminator = DiscriminatorTimeFreq(n_filters=self.cfg.MODEL.ARCHITECTURE.N_FILTERS)
#         # self.DiscriminatorFFT = Discriminator(n_filters=self.cfg.MODEL.ARCHITECTURE.N_FILTERS)
# 
#         self.optimizerG = optim.Adam(self.Generator.parameters(), lr=self.cfg.TRAIN.LR, betas=(self.cfg.TRAIN.BETA_1, 0.999))
#         self.optimizerD = optim.Adam(self.Discriminator.parameters(), lr=self.cfg.TRAIN.LR, betas=(self.cfg.TRAIN.BETA_1, 0.999))
#         # self.optimizerDFFT = optim.Adam(self.DiscriminatorFFT.parameters(), lr=self.cfg.TRAIN.LR, betas=(self.cfg.TRAIN.BETA_1, 0.999))
# 
#         self.lr_schedulerG = torch.optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=self.cfg.TRAIN.DECAY_ITERATIONS, gamma=self.cfg.TRAIN.DECAY_FACTOR)
#         self.lr_schedulerD = torch.optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=self.cfg.TRAIN.DECAY_ITERATIONS, gamma=self.cfg.TRAIN.DECAY_FACTOR)
# 
# 
# 
#         self.criterion = nn.BCELoss()
#         self.criterion_MSE = nn.MSELoss()
# 
#         def weights_init(m):
#             classname = m.__class__.__name__
#             if classname.find('Conv') != -1:
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
#             elif classname.find('BatchNorm') != -1:
#                 nn.init.normal_(m.weight.data, 1.0, 0.02)
#                 nn.init.constant_(m.bias.data, 0)
# 
#         self.Generator.apply(weights_init)
#         self.Discriminator.apply(weights_init)
#         # self.DiscriminatorFFT.apply(weights_init)
# 
#         self.DLoaderTrain = DataLoader(self.DatasetTrain, batch_size=self.minibatchsize, shuffle=True, num_workers=self.cfg.TRAIN.CPU_COUNT_LOADERS, drop_last=False)
# 
#         self.Discriminator.cuda(self.device)
#         # self.DiscriminatorFFT.cuda(self.device)
#         self.Generator.cuda(self.device)
# 
#     def do_epoch(self):
#         self.current_iteration = 0
#         print('----------------------------------')
#         print('VAE Epoch ' + str(self.current_epoch))
# 
# 
#         self.Discriminator.train()
#         # self.DiscriminatorFFT.train()
#         self.Generator.train()
# 
#         for k, x in enumerate(self.DLoaderTrain):
#             x, y, ystr = x
#             # break
# 
#             x = x.float().to(self.Discriminator.dummy_param.device)
#             x_orig_fft = torch.fft.fft(x.squeeze(1)).abs()#[:, :750]
#             x_orig_fft = torch.log10(x_orig_fft + 1e-12)
# 
#             # x = x.unsqueeze(1)
# 
#             self.Discriminator.zero_grad()
#             label = torch.full((x.shape[0] * 1,), 1.0, dtype=torch.float, device=self.Discriminator.dummy_param.device)
#             # label = torch.full((x.shape[0],), 1.0, dtype=torch.float, device=self.Discriminator.dummy_param.device)
#             output_time, output_freq = self.Discriminator((x.unsqueeze(1), x_orig_fft.unsqueeze(1)))#.view(-1)
#             loss_D_real_time = self.criterion(output_time.view(-1), label) * 0.8
#             loss_D_real_freq = self.criterion(output_freq.view(-1), label) * 0.2
#             loss_D_real = loss_D_real_time + loss_D_real_freq
#             loss_D_real.backward()
# 
#             # self.DiscriminatorFFT.zero_grad()
#             # label = torch.full((x_orig_fft.shape[0] * 3,), 1.0, dtype=torch.float, device=self.Discriminator.dummy_param.device)
#             # # label = torch.full((x.shape[0],), 1.0, dtype=torch.float, device=self.Discriminator.dummy_param.device)
#             # output = self.DiscriminatorFFT(x.unsqueeze(1)).view(-1)
#             # loss_DFFT_real = self.criterion(output, label)
#             # loss_DFFT_real.backward()
# 
#             noise = torch.randn(x.shape[0], self.cfg.MODEL.ARCHITECTURE.N_EMBEDDED_FEATURES, 3, device=self.Discriminator.dummy_param.device)
#             noise[:, :, 1] = noise[:, :, 0] + (0.5 * noise[:, :, 1])
#             noise[:, :, 2] = noise[:, :, 1] + (0.5 * noise[:, :, 2])
# 
#             fake = self.Generator(noise)#.detach()
#             x_fake_fft = torch.fft.fft(fake.squeeze(1)).abs()#[:, :750]
#             x_fake_fft = torch.log10(x_fake_fft + 1e-12)
# 
#             label.fill_(0.0)
#             output_time, output_freq = self.Discriminator((fake.detach(), x_fake_fft.detach().unsqueeze(1)))
#             loss_D_fake_time = self.criterion(output_time.view(-1), label) * 0.8
#             loss_D_fake_freq = self.criterion(output_freq.view(-1), label) * 0.2
#             loss_D_fake = loss_D_fake_time + loss_D_fake_freq
#             loss_D_fake.backward()
#             # output = self.DiscriminatorFFT(x_fake_fft.detach().unsqueeze(1)).view(-1)
#             # loss_DFFT_fake = self.criterion(output, label)
#             # loss_DFFT_fake.backward()
# 
#             self.optimizerD.step()
#             # self.optimizerDFFT.step()
# 
# 
#             self.Generator.zero_grad()
#             label.fill_(1.0)
#             output_time, output_freq = self.Discriminator((fake, x_fake_fft.unsqueeze(1)))
#             # outputFFT = self.DiscriminatorFFT(x_fake_fft.unsqueeze(1)).view(-1)
# 
#             loss_G1 = self.criterion(output_time.view(-1), label) * 0.8
#             loss_G2 = self.criterion(output_freq.view(-1), label) * 0.2
# 
# 
#             # x_orig_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0
#             # x_orig_fft[torch.isnan(x_fake_fft) | torch.isinf(x_fake_fft)] = 0
# 
# 
#             loss_G = loss_G1 + loss_G2
#             loss_G.backward()
#             self.optimizerG.step()
# 
#             losses = {
#                 'loss_D_fake_time': loss_D_fake_time.item(),
#                 'loss_D_real_time': loss_D_real_time.item(),
#                 'loss_D_fake_freq': loss_D_fake_freq.item(),
#                 'loss_D_real_freq': loss_D_real_freq.item(),
#                 # 'loss_DFFT_fake': loss_DFFT_fake.item(),
#                 # 'loss_DFFT_real': loss_DFFT_real.item(),
#                 'loss_G_time': loss_G1.item(),
#                 'loss_G_spect': loss_G2.item(),
#             }
# 
# 
#             if self.overall_iteration % self.cfg.TRAIN.SAVE_REPORT_ITERATION == 0 or k==0:
#                 self.print_losses_to_file(losses)
#                 self.plot_to_file(fake, x)
# 
#             if self.overall_iteration % self.cfg.TRAIN.SAVE_MODEL_ITERATION == 0 or k==0:
#                 self.save_model()
# 
#             self.current_iteration += 1
#             self.overall_iteration += 1
# 
#         self.current_epoch += 1
#         self.lr_schedulerD.step()
#         self.lr_schedulerG.step()
# 
# 
#     def print_losses_to_file(self, losses):
#         path = self.path_report_losses
#         lr = self.optimizerG.param_groups[0]['lr']
#         #f.write('Epoch, lr, lossG_fake, lossG_real, lossD\n')
#         printStr = f'{self.current_epoch}, {lr}, ' + ', '.join([f'{i:.8f}' for k, i in losses.items()]) + '\n'
#         print(self.overall_iteration, self.current_epoch, self.current_iteration, losses)
#         with open(path, 'a') as f:
#             f.write(printStr)
# 
# 
#     def plot_to_file(self, fake, real):
#         x_real = real[-1].detach().cpu().squeeze().numpy()
#         x_fake = fake[-1].detach().cpu().squeeze().numpy()
#         img_path = os.path.join(self.path_report_images, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}")
#         img_path_spect = os.path.join(self.path_report_images, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_spect")
# 
#         plt.figure(figsize=(12, 4))
#         ax0 = plt.subplot(2, 1, 1)
#         plt.plot(x_real)
#         plt.title('Real')
#         plt.subplot(2, 1, 2, sharex=ax0)
#         plt.plot(x_fake)
#         plt.title('Fake')
#         plt.savefig(img_path + '.png')
#         plt.savefig(img_path + '.svg')
#         plt.close()
# 
#         f, Pxx = PSD(fake.detach().cpu().numpy().squeeze(), 500)
#         Pxx_fake = Pxx.mean(axis=0)
# 
#         f, Pxx = PSD(real.detach().cpu().numpy().squeeze(), 500)
#         Pxx_real = Pxx.mean(axis=0)
# 
#         plt.figure(figsize=(12, 4))
#         plt.semilogy(f[f<100], Pxx_real[f<100])
#         plt.semilogy(f[f<100], Pxx_fake[f<100])
#         plt.legend(['Real', 'Fake'])
#         plt.savefig(img_path_spect + '.png')
#         plt.savefig(img_path_spect + '.svg')
#         plt.close()
#         #plt.show()
# 
#     def save_model(self):
#         PATH_D = os.path.join(self.path_report_models, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_Discriminator")
#         PATH_G = os.path.join(self.path_report_models, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_Generator")
#         torch.save(self.Discriminator.state_dict(), PATH_D)
#         torch.save(self.Generator.state_dict(), PATH_G)
# 
#     def train(self):
#         for k in range(self.epochs):
#             self.do_epoch()
