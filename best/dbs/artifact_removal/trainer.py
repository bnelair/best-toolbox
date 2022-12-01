import yaml

from datetime import datetime



import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from best.dbs.artifact_removal.model import *
from best.dbs.artifact_removal.dataset import *
from best.dbs import *
from best.dbs.artifact_removal.model import dbs_artifact_removal_network_v2

from best._config import ObjDictToDict

torch.autograd.set_detect_anomaly(True)

class TrainerUpgrade:
    def __init__(self, config):
        self.cfg = config
        self.model_name = self.cfg.NAME
        self.fs = self.cfg.MODEL.ARCHITECTURE.FS
        exec(
        f"self.DatasetTrain={self.cfg.TRAIN.DATASET.split('.')[-1]}({self.cfg.TRAIN.SIGNAL_LENGTH}, {self.cfg.TRAIN.EEG_GENERATORS_DCGAN}, {self.fs}, {self.cfg.TRAIN.STIM_ARTIFACT})"
        )
        self.DatasetTrain._len = self.cfg.TRAIN.BATCH_SIZE * self.cfg.TRAIN.N_ITERATIONS

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
            f.write('Epoch, lr, loss_rec_signal, loss_rec_spectrogram, loss_detection\n')

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

        self.model = dbs_artifact_removal_network_v2()
        self.model.cuda(self.GPUs[0])

        # exec(
        # f"self.model={self.cfg.MODEL.MODEL.split('.')[-1]}(n_filters={self.cfg.MODEL.ARCHITECTURE.N_FILTERS})"
        # )


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.BASE_LR, betas=(self.cfg.TRAIN.BETA_1, 0.999))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.TRAIN.DECAY_ITERATIONS, gamma=self.cfg.TRAIN.DECAY_FACTOR)

        self.criterionMSE = nn.MSELoss()
        self.criterionBCE = nn.BCELoss()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        # self.model.apply(weights_init)
        self.DLoaderTrain = DataLoader(self.DatasetTrain, batch_size=self.minibatchsize, shuffle=True, num_workers=self.cfg.TRAIN.CPU_COUNT_LOADERS, drop_last=False)
        # self.model.cuda(self.device)


    def do_epoch(self):
        self.current_iteration = 0
        print('----------------------------------')
        print('VAE Epoch ' + str(self.current_epoch))


        self.model.train()
        for k, (x_orig, x_art, y_art) in enumerate(self.DLoaderTrain):
            # break
            self.model.zero_grad()
            x_orig = x_orig.float().detach().to(self.device)
            x_art = x_art.float().detach().to(self.device)
            y_art = y_art.float().detach().to(self.device)

            x_art_diff_1 = x_art.diff(1, -1, x_art[:, :, 0].view(x_art.shape[0], x_art.shape[1], 1))
            x_art_diff_2 = x_art_diff_1.diff(1, -1, x_art_diff_1[:, :, 0].view(x_art_diff_1.shape[0], x_art_diff_1.shape[1], 1))

            x_art_inp = torch.cat(
                (
                    x_art,
                    x_art_diff_1,
                    x_art_diff_2
                ), 1
            )

            x_rec, att = self.model(x_art_inp)
            #x_orig_fft = torch.stft(x_orig.squeeze(1), n_fft=2*500, hop_length=100, return_complex=True, ).abs()
            #x_rec_fft = torch.stft(x_rec.squeeze(1), n_fft=2*500, hop_length=100, return_complex=True, ).abs()
            # break
            x_orig_fft = torch.fft.fft(x_orig.squeeze(1)).abs()
            x_rec_fft = torch.fft.fft(x_rec.squeeze(1)).abs()

            x_orig_fft = torch.log10(x_orig_fft+0.0001)
            x_rec_fft = torch.log10(x_rec_fft+0.0001)

            x_rec_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0
            x_orig_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0

            x_orig_fft[torch.isnan(x_rec_fft) | torch.isinf(x_rec_fft)] = 0
            x_rec_fft[torch.isnan(x_rec_fft) | torch.isinf(x_rec_fft)] = 0


            #att[att < 0] = att[att < 0]*0
            #att[att > 1] = att[att > 1]/att[att > 1]


            loss_signal = self.criterionMSE(x_rec[~torch.isnan(x_rec)], x_orig[~torch.isnan(x_rec)]) * self.cfg.TRAIN.WEIGHT_SIGNAL_RECONSTRUCTION
            loss_fft = self.criterionMSE(x_rec_fft[~torch.isnan(x_rec_fft)], x_orig_fft[~torch.isnan(x_rec_fft)])* self.cfg.TRAIN.WEIGHT_STFT_RECONSTRUCTION
            loss_att = self.criterionBCE(att[~torch.isnan(att)], y_art[~torch.isnan(att)]) * self.cfg.TRAIN.WEIGHT_DETECTION


            loss = loss_signal + loss_fft #+ loss_att
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.lr_scheduler.step()


            losses = {
                'loss_rec_signal': loss_signal.item(),
                'loss_rec_spectrogram': loss_fft.item(),
                'loss_detection': loss_att.item(),
            }


            if self.overall_iteration % self.cfg.TRAIN.SAVE_REPORT_ITERATION == 0 or k==0:
                self.print_losses_to_file(losses)
                self.plot_to_file(x_orig, x_art, x_rec, y_art, att)

            if self.overall_iteration % self.cfg.TRAIN.SAVE_MODEL_ITERATION == 0 or k==0:
                self.save_model()

            self.current_iteration += 1
            self.overall_iteration += 1
        self.current_epoch += 1


    def print_losses_to_file(self, losses):
        path = self.path_report_losses
        lr = self.optimizer.param_groups[0]['lr']
        printStr = f'{self.current_iteration}, {lr}, ' + ', '.join([f'{i:.8f}' for k, i in losses.items()]) + '\n'
        print(self.overall_iteration, self.current_epoch, self.current_iteration, losses)
        with open(path, 'a') as f:
            f.write(printStr)

    # (x_orig, x_rec, y_art, att)
    def plot_to_file(self, x_orig, x_art, x_rec, y_art, yy_art):
        x_orig = x_orig[-1].detach().cpu().squeeze().numpy()
        x_rec = x_rec[-1].detach().cpu().squeeze().numpy()
        x_art = x_art[-1].detach().cpu().squeeze().numpy()
        y_art = y_art[-1].detach().cpu().squeeze().numpy()
        yy_art = yy_art[-1].detach().cpu().squeeze().numpy()
        img_path = os.path.join(self.path_report_images, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}")

        plt.figure(figsize=(12, 4))
        ax0 = plt.subplot(3, 1, 1)
        plt.plot(x_orig)
        plt.plot(x_rec)
        plt.subplot(3, 1, 2, sharex=ax0)
        plt.plot(x_art)
        plt.subplot(3, 1, 3, sharex=ax0)
        plt.plot(y_art)
        plt.plot(yy_art)
        plt.savefig(img_path + '.png')
        # plt.savefig(img_path + '.pdf')
        plt.close()
        plt.show()

    def save_model(self):
        PATH = os.path.join(self.path_report_models, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_ArtifactEraser.pt")
        # torch.save(self.model.state_dict(), PATH)

    def train(self):
        self.do_epoch()


class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.model_name = self.cfg.NAME
        self.fs = self.cfg.MODEL.ARCHITECTURE.FS
        exec(
        f"self.DatasetTrain={self.cfg.TRAIN.DATASET.split('.')[-1]}({self.cfg.TRAIN.SIGNAL_LENGTH}, {self.cfg.TRAIN.EEG_GENERATORS_DCGAN}, {self.fs}, {self.cfg.TRAIN.STIM_ARTIFACT})"
        )
        self.DatasetTrain._len = self.cfg.TRAIN.BATCH_SIZE * self.cfg.TRAIN.N_ITERATIONS

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
            f.write('Epoch, lr, loss_rec_signal, loss_rec_spectrogram, loss_detection\n')

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

        #self.model = GRU_Denoiser(n_filters=self.cfg.MODEL.ARCHITECTURE.N_FILTERS)

        exec(
        f"self.model={self.cfg.MODEL.MODEL.split('.')[-1]}(n_filters={self.cfg.MODEL.ARCHITECTURE.N_FILTERS})"
        )


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.BASE_LR, betas=(self.cfg.TRAIN.BETA_1, 0.999))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.TRAIN.DECAY_ITERATIONS, gamma=self.cfg.TRAIN.DECAY_FACTOR)

        self.criterionMSE = nn.MSELoss()
        self.criterionBCE = nn.BCELoss()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.model.apply(weights_init)
        self.DLoaderTrain = DataLoader(self.DatasetTrain, batch_size=self.minibatchsize, shuffle=True, num_workers=self.cfg.TRAIN.CPU_COUNT_LOADERS, drop_last=False)
        self.model.cuda(self.device)


    def do_epoch(self):
        self.current_iteration = 0
        print('----------------------------------')
        print('VAE Epoch ' + str(self.current_epoch))


        self.model.train()
        #print('DOPICE')
        for k, (x_orig, x_art, y_art) in enumerate(self.DLoaderTrain):
            #break
            self.model.zero_grad()
            x_orig = x_orig.float().detach().to(self.device)
            x_art = x_art.float().detach().to(self.device)
            y_art = y_art.float().detach().to(self.device)

            x_rec, att = self.model(x_art)
            #x_orig_fft = torch.stft(x_orig.squeeze(1), n_fft=2*500, hop_length=100, return_complex=True, ).abs()
            #x_rec_fft = torch.stft(x_rec.squeeze(1), n_fft=2*500, hop_length=100, return_complex=True, ).abs()

            x_orig_fft = torch.fft.fft(x_orig.squeeze(1)).abs()
            x_rec_fft = torch.fft.fft(x_rec.squeeze(1)).abs()

            x_orig_fft = torch.log10(x_orig_fft+0.0001)
            x_rec_fft = torch.log10(x_rec_fft+0.0001)

            x_rec_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0
            x_orig_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0

            x_orig_fft[torch.isnan(x_rec_fft) | torch.isinf(x_rec_fft)] = 0
            x_rec_fft[torch.isnan(x_rec_fft) | torch.isinf(x_rec_fft)] = 0


            #att[att < 0] = att[att < 0]*0
            #att[att > 1] = att[att > 1]/att[att > 1]


            loss_signal = self.criterionMSE(x_rec[~torch.isnan(x_rec)], x_orig[~torch.isnan(x_rec)]) * self.cfg.TRAIN.WEIGHT_SIGNAL_RECONSTRUCTION
            loss_fft = self.criterionMSE(x_rec_fft[~torch.isnan(x_rec_fft)], x_orig_fft[~torch.isnan(x_rec_fft)])* self.cfg.TRAIN.WEIGHT_STFT_RECONSTRUCTION
            loss_att = self.criterionBCE(att[~torch.isnan(att)], y_art[~torch.isnan(att)]) * self.cfg.TRAIN.WEIGHT_DETECTION
            # loss_att = self.criterionMSE(x_art[~torch.isnan(x_rec)], att[~torch.isnan(x_rec)]) * self.cfg.TRAIN.WEIGHT_DETECTION


            loss = loss_signal + loss_att + loss_fft
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.lr_scheduler.step()


            losses = {
                'loss_rec_signal': loss_signal.item(),
                'loss_rec_spectrogram': loss_fft.item(),
                'loss_detection': loss_att.item(),
            }


            if self.overall_iteration % self.cfg.TRAIN.SAVE_REPORT_ITERATION == 0 or k==0:
                self.print_losses_to_file(losses)
                self.plot_to_file(x_orig, x_art, x_rec, y_art, att)

            if self.overall_iteration % self.cfg.TRAIN.SAVE_MODEL_ITERATION == 0 or k==0:
                self.save_model()

            self.current_iteration += 1
            self.overall_iteration += 1

        self.current_epoch += 1


    def print_losses_to_file(self, losses):
        path = self.path_report_losses
        lr = self.optimizer.param_groups[0]['lr']
        printStr = f'{self.current_iteration}, {lr}, ' + ', '.join([f'{i:.8f}' for k, i in losses.items()]) + '\n'
        print(self.overall_iteration, self.current_epoch, self.current_iteration, losses)
        with open(path, 'a') as f:
            f.write(printStr)

    # (x_orig, x_rec, y_art, att)
    def plot_to_file(self, x_orig, x_art, x_rec, y_art, yy_art):
        x_orig = x_orig[-1].detach().cpu().squeeze().numpy()
        x_rec = x_rec[-1].detach().cpu().squeeze().numpy()
        x_art = x_art[-1].detach().cpu().squeeze().numpy()
        y_art = y_art[-1].detach().cpu().squeeze().numpy()
        yy_art = yy_art[-1].detach().cpu().squeeze().numpy()
        img_path = os.path.join(self.path_report_images, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}")

        plt.figure(figsize=(12, 4))
        ax0 = plt.subplot(3, 1, 1)
        plt.plot(x_orig)
        plt.plot(x_rec)
        plt.subplot(3, 1, 2, sharex=ax0)
        plt.plot(x_art)
        plt.subplot(3, 1, 3, sharex=ax0)
        plt.plot(y_art)
        plt.plot(yy_art)
        plt.savefig(img_path + '.png')
        plt.savefig(img_path + '.pdf')
        plt.close()
        #plt.show()

    def save_model(self):
        PATH = os.path.join(self.path_report_models, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_ArtifactEraser.pt")
        torch.save(self.model.state_dict(), PATH)

    def train(self):
        self.do_epoch()


class TrainerLight:
    def __init__(self, config):
        self.cfg = config
        self.model_name = self.cfg.NAME
        exec(
        f"self.DatasetTrain={self.cfg.TRAIN.DATASET.split('.')[-1]}({self.cfg.TRAIN.SIGNAL_LENGTH}, {self.cfg.TRAIN.EEG_GENERATORS_DCGAN}, {self.cfg.TRAIN.STIM_ARTIFACT})"
        )
        self.DatasetTrain._len = self.cfg.TRAIN.BATCH_SIZE * self.cfg.TRAIN.N_ITERATIONS


        self.path_report = os.path.join(self.cfg.TRAIN.PATH_REPORT, self.cfg.NAME + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.path_report_models = os.path.join(self.path_report, 'Models')
        self.path_report_images = os.path.join(self.path_report, 'Images')
        self.path_report_config = os.path.join(self.path_report, 'config.yaml')
        self.path_report_losses = os.path.join(self.path_report, 'losses.csv')
        self.path_report_losses_validation = os.path.join(self.path_report, 'losses_val.csv')


        # os.mkdir(self.path_report)
        # os.mkdir(self.path_report_models)
        # os.mkdir(self.path_report_images)
        # with open(self.path_report_config, 'w') as file:
            # yaml.dump(ObjDictToDict(self.cfg), file)


        # with open(self.path_report_losses, 'w') as f:
        #     f.write('Epoch, lr, loss_rec_signal, loss_rec_spectrogram, loss_detection\n')

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

        self.model = dbs_artifact_removal_network(n_filters=self.cfg.MODEL.ARCHITECTURE.N_FILTERS)

        exec(
        f"self.model={self.cfg.MODEL.MODEL.split('.')[-1]}(n_filters={self.cfg.MODEL.ARCHITECTURE.N_FILTERS})"
        )


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.BASE_LR, betas=(self.cfg.TRAIN.BETA_1, 0.999))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.TRAIN.DECAY_ITERATIONS, gamma=self.cfg.TRAIN.DECAY_FACTOR)

        self.criterionMSE = nn.MSELoss()
        self.criterionBCE = nn.BCELoss()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.model.apply(weights_init)
        self.DLoaderTrain = DataLoader(self.DatasetTrain, batch_size=self.minibatchsize, shuffle=True, num_workers=self.cfg.TRAIN.CPU_COUNT_LOADERS, drop_last=False)
        self.model.cuda(self.device)


    def do_epoch(self):
        self.current_iteration = 0
        print('----------------------------------')
        print('VAE Epoch ' + str(self.current_epoch))


        self.model.train()
        #print('DOPICE')
        for k, (x_orig, x_art, y_art) in enumerate(self.DLoaderTrain):
            #break
            self.model.zero_grad()
            x_orig = x_orig.float().detach().to(self.device)
            x_art = x_art.float().detach().to(self.device)
            y_art = y_art.float().detach().to(self.device)

            x_rec = self.model(x_art)
            #x_orig_fft = torch.stft(x_orig.squeeze(1), n_fft=2*500, hop_length=100, return_complex=True, ).abs()
            #x_rec_fft = torch.stft(x_rec.squeeze(1), n_fft=2*500, hop_length=100, return_complex=True, ).abs()

            x_orig_fft = torch.fft.fft(x_orig.squeeze(1)).abs()
            x_rec_fft = torch.fft.fft(x_rec.squeeze(1)).abs()

            x_orig_fft = torch.log10(x_orig_fft+0.0001)
            x_rec_fft = torch.log10(x_rec_fft+0.0001)

            x_rec_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0
            x_orig_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0

            x_orig_fft[torch.isnan(x_rec_fft) | torch.isinf(x_rec_fft)] = 0
            x_rec_fft[torch.isnan(x_rec_fft) | torch.isinf(x_rec_fft)] = 0


            #att[att < 0] = att[att < 0]*0
            #att[att > 1] = att[att > 1]/att[att > 1]


            loss_signal = self.criterionMSE(x_rec[~torch.isnan(x_rec)], x_orig[~torch.isnan(x_rec)]) * self.cfg.TRAIN.WEIGHT_SIGNAL_RECONSTRUCTION
            loss_fft = self.criterionMSE(x_rec_fft[~torch.isnan(x_rec_fft)], x_orig_fft[~torch.isnan(x_rec_fft)])* self.cfg.TRAIN.WEIGHT_STFT_RECONSTRUCTION


            loss = loss_signal + loss_fft
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.lr_scheduler.step()


            losses = {
                'loss_rec_signal': loss_signal.item(),
                'loss_rec_spectrogram': loss_fft.item(),
            }


            if self.overall_iteration % self.cfg.TRAIN.SAVE_REPORT_ITERATION == 0 or k==0:
                self.print_losses_to_file(losses)
                self.plot_to_file(x_orig, x_art, x_rec, y_art)

            if self.overall_iteration % self.cfg.TRAIN.SAVE_MODEL_ITERATION == 0 or k==0:
                self.save_model()

            self.current_iteration += 1
            self.overall_iteration += 1

        self.current_epoch += 1


    def print_losses_to_file(self, losses):
        path = self.path_report_losses
        lr = self.optimizer.param_groups[0]['lr']
        printStr = f'{self.current_iteration}, {lr}, ' + ', '.join([f'{i:.8f}' for k, i in losses.items()]) + '\n'
        print(self.overall_iteration, self.current_epoch, self.current_iteration, losses)
        with open(path, 'a') as f:
            f.write(printStr)

    # (x_orig, x_rec, y_art, att)
    def plot_to_file(self, x_orig, x_art, x_rec, y_art):
        x_orig = x_orig[-1].detach().cpu().squeeze().numpy()
        x_rec = x_rec[-1].detach().cpu().squeeze().numpy()
        x_art = x_art[-1].detach().cpu().squeeze().numpy()
        y_art = y_art[-1].detach().cpu().squeeze().numpy()
        img_path = os.path.join(self.path_report_images, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}")

        plt.figure(figsize=(12, 4))
        ax0 = plt.subplot(3, 1, 1)
        plt.plot(x_orig)
        plt.plot(x_rec)
        plt.subplot(3, 1, 2, sharex=ax0)
        plt.plot(x_art)
        plt.subplot(3, 1, 3, sharex=ax0)
        plt.plot(y_art)
        plt.savefig(img_path + '.png')
        plt.savefig(img_path + '.svg')
        plt.close()
        #plt.show()

    def save_model(self):
        PATH = os.path.join(self.path_report_models, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_ArtifactEraser.pt")
        torch.save(self.model.state_dict(), PATH)

    def train(self):
        self.do_epoch()


class Trainer_:
    def __init__(self, config):
        self.cfg = config
        self.model_name = self.cfg.NAME
        exec(
        f"self.DatasetTrain={self.cfg.TRAIN.DATASET.split('.')[-1]}({self.cfg.TRAIN.SIGNAL_LENGTH}, {self.cfg.TRAIN.EEG_GENERATORS_DCGAN}, {self.cfg.TRAIN.STIM_ARTIFACT})"
        )
        self.DatasetTrain._len = self.cfg.TRAIN.BATCH_SIZE * self.cfg.TRAIN.N_ITERATIONS


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
            f.write('Epoch, lr, loss_rec_signal, loss_rec_spectrogram, loss_detection\n')

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

        #self.model = GRU_Denoiser(n_filters=self.cfg.MODEL.ARCHITECTURE.N_FILTERS)

        exec(
        f"self.model={self.cfg.MODEL.MODEL.split('.')[-1]}(n_filters={self.cfg.MODEL.ARCHITECTURE.N_FILTERS})"
        )


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.BASE_LR, betas=(self.cfg.TRAIN.BETA_1, 0.999))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.TRAIN.DECAY_ITERATIONS, gamma=self.cfg.TRAIN.DECAY_FACTOR)

        self.criterionMSE = nn.MSELoss()
        self.criterionBCE = nn.BCELoss()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.model.apply(weights_init)
        self.DLoaderTrain = DataLoader(self.DatasetTrain, batch_size=self.minibatchsize, shuffle=True, num_workers=self.cfg.TRAIN.CPU_COUNT_LOADERS, drop_last=False)
        self.model.cuda(self.device)


    def do_epoch(self):
        self.current_iteration = 0
        print('----------------------------------')
        print('VAE Epoch ' + str(self.current_epoch))


        self.model.train()
        #print('DOPICE')
        for k, (x_orig, x_art, y_art) in enumerate(self.DLoaderTrain):
            #break
            self.model.zero_grad()
            x_orig = x_orig.float().detach().to(self.device)
            x_art = x_art.float().detach().to(self.device)
            y_art = y_art.float().detach().to(self.device)

            x_rec, att = self.model(x_art)
            #x_orig_fft = torch.stft(x_orig.squeeze(1), n_fft=2*500, hop_length=100, return_complex=True, ).abs()
            #x_rec_fft = torch.stft(x_rec.squeeze(1), n_fft=2*500, hop_length=100, return_complex=True, ).abs()

            x_orig_fft = torch.fft.fft(x_orig.squeeze(1)).abs()
            x_rec_fft = torch.fft.fft(x_rec.squeeze(1)).abs()

            x_orig_fft = torch.log10(x_orig_fft+0.0001)
            x_rec_fft = torch.log10(x_rec_fft+0.0001)

            x_rec_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0
            x_orig_fft[torch.isnan(x_orig_fft) | torch.isinf(x_orig_fft)] = 0

            x_orig_fft[torch.isnan(x_rec_fft) | torch.isinf(x_rec_fft)] = 0
            x_rec_fft[torch.isnan(x_rec_fft) | torch.isinf(x_rec_fft)] = 0


            #att[att < 0] = att[att < 0]*0
            #att[att > 1] = att[att > 1]/att[att > 1]


            loss_signal = self.criterionMSE(x_rec[~torch.isnan(x_rec)], x_orig[~torch.isnan(x_rec)]) * self.cfg.TRAIN.WEIGHT_SIGNAL_RECONSTRUCTION
            loss_fft = self.criterionMSE(x_rec_fft[~torch.isnan(x_rec_fft)], x_orig_fft[~torch.isnan(x_rec_fft)])* self.cfg.TRAIN.WEIGHT_STFT_RECONSTRUCTION
            loss_att = self.criterionBCE(att[~torch.isnan(att)], y_art[~torch.isnan(att)]) * self.cfg.TRAIN.WEIGHT_DETECTION


            loss = loss_signal + loss_att + loss_fft
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.lr_scheduler.step()


            losses = {
                'loss_rec_signal': loss_signal.item(),
                'loss_rec_spectrogram': loss_fft.item(),
                'loss_detection': loss_att.item(),
            }


            if self.overall_iteration % self.cfg.TRAIN.SAVE_REPORT_ITERATION == 0 or k==0:
                self.print_losses_to_file(losses)
                self.plot_to_file(x_orig, x_art, x_rec, y_art, att)

            if self.overall_iteration % self.cfg.TRAIN.SAVE_MODEL_ITERATION == 0 or k==0:
                self.save_model()

            self.current_iteration += 1
            self.overall_iteration += 1

        self.current_epoch += 1


    def print_losses_to_file(self, losses):
        path = self.path_report_losses
        lr = self.optimizer.param_groups[0]['lr']
        printStr = f'{self.current_iteration}, {lr}, ' + ', '.join([f'{i:.8f}' for k, i in losses.items()]) + '\n'
        print(self.overall_iteration, self.current_epoch, self.current_iteration, losses)
        with open(path, 'a') as f:
            f.write(printStr)

    # (x_orig, x_rec, y_art, att)
    def plot_to_file(self, x_orig, x_art, x_rec, y_art, yy_art):
        x_orig = x_orig[-1].detach().cpu().squeeze().numpy()
        x_rec = x_rec[-1].detach().cpu().squeeze().numpy()
        x_art = x_art[-1].detach().cpu().squeeze().numpy()
        y_art = y_art[-1].detach().cpu().squeeze().numpy()
        yy_art = yy_art[-1].detach().cpu().squeeze().numpy()
        img_path = os.path.join(self.path_report_images, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}")

        plt.figure(figsize=(12, 4))
        ax0 = plt.subplot(3, 1, 1)
        plt.plot(x_orig)
        plt.plot(x_rec)
        plt.subplot(3, 1, 2, sharex=ax0)
        plt.plot(x_art)
        plt.subplot(3, 1, 3, sharex=ax0)
        plt.plot(y_art)
        plt.plot(yy_art)
        # plt.savefig(img_path + '.png')
        plt.savefig(img_path + '.svg')
        plt.close()
        #plt.show()

    def save_model(self):
        PATH = os.path.join(self.path_report_models, f"{self.cfg.NAME}_epoch_{self.current_epoch:05d}_iteration_{self.current_iteration:05d}_step_{self.overall_iteration:05d}_ArtifactEraser.pt")
        torch.save(self.model.state_dict(), PATH)

    def train(self):
        self.do_epoch()



