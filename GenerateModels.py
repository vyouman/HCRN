# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from istft_irfft import istft_irfft
from utils import concatenateFeature

def init_weights(m):
    classname = m.__class__.__name__
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    # elif isinstance(m, nn.BatchNorm2d):
    elif classname.find('BatchNorm') != -1:
        # m.weight.data.fill_(1)
        m.bias.data.zero_()
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
    else:  # for lstm
        for p in m.parameters():
            if len(p.shape) >= 2:
                nn.init.orthogonal_(p.data)
            else:
                nn.init.normal_(p.data)


class convBlock(nn.Module):
    def __init__(self, numIn, numOut, kernel_size=3, dilation_rate=1):
        super(convBlock, self).__init__()
        # keep the same
        if dilation_rate == 1:
            self.sc = nn.Sequential(
                nn.BatchNorm2d(numIn),
                nn.ReLU(),
                nn.Conv2d(numIn, numOut // 2, kernel_size, 1, kernel_size // 2), # (10, 32, 128, 128)

                nn.BatchNorm2d(numOut // 2),
                nn.ReLU(),
                nn.Conv2d(numOut // 2, numOut // 2, kernel_size, 1, kernel_size // 2), # (10, 32, 128, 128)

                nn.BatchNorm2d(numOut // 2),
                nn.ReLU(),
                nn.Conv2d(numOut // 2, numOut, 1, 1)
        )
        else:
            self.sc = nn.Sequential(
                nn.BatchNorm2d(numIn),
                nn.ReLU(),
                nn.Conv2d(numIn, numOut // 2, kernel_size, 1, (kernel_size + (kernel_size - 1) * (dilation_rate - 1)) // 2, dilation_rate),

                nn.BatchNorm2d(numOut // 2),
                nn.ReLU(),
                nn.Conv2d(numOut // 2, numOut // 2, kernel_size, 1, (kernel_size + (kernel_size - 1 ) * (dilation_rate -1 )) // 2, dilation_rate),

                nn.BatchNorm2d(numOut // 2),
                nn.ReLU(),
                nn.Conv2d(numOut // 2, numOut, 1, 1)
            )

    def forward(self, x):
        x = self.sc(x)
        return x

class convBlock_noAct(nn.Module):
    def __init__(self, numIn, numOut, kernel_size=3, dilation_rate=1):
        super(convBlock_noAct, self).__init__()
        # keep the same
        if dilation_rate == 1:
            self.sc = nn.Sequential(
                nn.BatchNorm2d(numIn),
                nn.Conv2d(numIn, numOut // 2, kernel_size, 1, kernel_size // 2), # (10, 32, 128, 128)

                nn.BatchNorm2d(numOut // 2),
                nn.Conv2d(numOut // 2, numOut // 2, kernel_size, 1, kernel_size // 2), # (10, 32, 128, 128)

                nn.BatchNorm2d(numOut // 2),
                nn.Conv2d(numOut // 2, numOut, 1, 1)
        )
        else:
            self.sc = nn.Sequential(
                nn.BatchNorm2d(numIn),
                nn.Conv2d(numIn, numOut // 2, kernel_size, 1, kernel_size // 2),  # (10, 32, 128, 128)

                nn.BatchNorm2d(numOut // 2),
                nn.Conv2d(numOut // 2, numOut // 2, kernel_size, 1, (kernel_size + (kernel_size - 1 ) * (dilation_rate -1 )) // 2, dilation_rate),

                nn.BatchNorm2d(numOut // 2),
                nn.Conv2d(numOut // 2, numOut, 1, 1)
            )

    def forward(self, x):
        x = self.sc(x)
        return x


class skipLayer(nn.Module):
    def __init__(self, numIn, numOut):
        super(skipLayer, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.conv = nn.Conv2d(numIn, numOut, 1, 1)

    def forward(self, x):
        if self.numIn == self.numOut:
            return x
        else:
            x = self.conv(x)
            return x

class Residual(nn.Module):
    def __init__(self, numIn, numOut, kernel_size=3, dilation_rate=1, noAct=0):
        super(Residual, self).__init__()
        if not noAct:
            self.conv = convBlock(numIn, numOut, kernel_size, dilation_rate)
        else:
            self.conv = convBlock_noAct(numIn, numOut, kernel_size, dilation_rate)
        self.skip_layer = skipLayer(numIn, numOut)

    def forward(self, x):
        c = self.conv(x)
        s = self.skip_layer(x)
        out = c + s
        return out

class Hourglass_lstm(nn.Module):
    def __init__(self, n, numIn, numOut, pool_free=0, kernel_size=3, dilation_rate=1, noAct=0):
        super(Hourglass_lstm, self).__init__()
        self.upper = Residual(numIn, numOut, kernel_size, dilation_rate, noAct)
        if pool_free:
            self.maxpool = nn.Conv2d(numIn, numIn, 2, 2)
            self.upsampling = nn.ConvTranspose2d(numOut, numOut, 2, 2)
        else:
            self.maxpool = nn.MaxPool2d(2, 2)
            self.upsampling = nn.Upsample(scale_factor=2) 
        self.lower = Residual(numIn, numOut, kernel_size, dilation_rate, noAct)

        self.n = n

        if n == 1:
            # LSTM
            self.LSTM = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)


        if n > 1:
            self.hourglass = Hourglass_lstm(n - 1, numOut, numOut, pool_free, kernel_size, dilation_rate, noAct)

        self.res = Residual(numOut, numOut, kernel_size, dilation_rate, noAct)

    def forward(self, x):
        # print('Shape of input', x.shape)
        up1 = self.upper(x)
        # print('Shape of input after upper up1', up1.shape)
        low1 = self.maxpool(x)
        # print('Shape of input after maxpool', low1.shape)
        low1 = self.lower(low1)
        # print('Shape of input after lower1', low1.shape)

        if self.n > 1:
            low2 = self.hourglass(low1)
            low2 = self.res(low2)
        else:
            # reshape
            low1 = low1.permute(0, 2, 1, 3)  # b, t, c, f
            low1 = low1.reshape(low1.size()[0], low1.size()[1], -1)  # b, t, c*f
            # lstm
            low1, (hn, cn) = self.LSTM(low1) # b, t, 1024
            # reshape
            low1 = low1.reshape(low1.size()[0], low1.size()[1], 64, -1) # b, t, c, f
            low1 = low1.permute(0, 2, 1, 3)
            low2 = self.res(low1)

        # print('Shape of low2,', low2.shape)
        up2 = self.upsampling(low2)
        # print('Shape of up2,', up2.shape)

        out = up1 + up2

        return out

class EncoderNetLinear_lstm(nn.Module):
    def __init__(self, num_freq, num_mels, useAct='Sigmoid', n=3,
                 outChannel=64, num_frames=128, pool_free=0, kernel_size=3, dilation_rate=1):

        super(EncoderNetLinear_lstm, self).__init__()

        self.outChannel = outChannel

        self.hg_net = Hourglass_lstm(n, 1, self.outChannel, pool_free, kernel_size, dilation_rate)  
        self.output_res = Residual(self.outChannel + 1, self.outChannel, kernel_size,
                                   dilation_rate)  
        self.output_conv = nn.Conv2d(self.outChannel, 1, 3, 1, 1)  
        self.useAct = useAct
        if useAct == 'Sigmoid':
            self.output_act = nn.Sigmoid()
        elif useAct == 'Tanh': 
            self.output_act = nn.Tanh()

        self._initialize_weights()

    def reshapeForCNN(self, x):
        (B, num_frames, num_freq) = x.shape
        x = x.reshape(B, -1, num_frames, num_freq)
        return x

    def forward(self, input1):
        '''
        :param input1: (B, num_frame, num_freq)
        :return: output_linear
        '''
        input1 = self.reshapeForCNN(input1)  # (1, num_frames, num_freq)
        x = self.hg_net(input1)
        # print('Shape of hg_output after hg_net', hg_output.shape)
        x = concatenateFeature([x, input1], dim=1)  # (#, num_frames, num_freq)
        # print('Shape of hg_output_linear after concatenating features', hg_output_linear.shape)
        x = self.output_res(x)
        x = self.output_conv(x)
        if self.useAct is not None:
            x = self.output_act(x)  # (N, 1, T, H)
        (N, _, T, H) = x.shape
        x = x.reshape(N, T, H)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            init_weights(m)


class WaveLoss(nn.Module):
    def __init__(self, dBscale = 1, denormalize=1, max_db=100, ref_db=20, nfft=256, hop_size=128, clip=1):
        super(WaveLoss, self).__init__()
        self.dBscale = dBscale
        self.denormalize = denormalize
        self.max_db = max_db
        self.ref_db = ref_db
        self.nfft = nfft
        self.hop_size = hop_size
        self.loss = nn.L1Loss()
        self.clip = clip

    def genWav(self, S, phase):
        '''
        :param S: (B, F, T) to be padded with 0 in this function
        :param phase: (B, F, T)
        :return: (B, num_samples)
        '''
        if self.dBscale:
            if self.denormalize:
                # denormalization
                S = S * self.max_db - self.max_db + self.ref_db
            S = 10 ** (S * 0.05)

        # pad with 0
        B, F, T = S.shape

        if self.clip:
            pad = torch.zeros(B, 1, T).to(S.device)
            Sfull = concatenateFeature([S, pad], dim=-2)
        else:
            Sfull = S

        # deal with the complex
        Sfull_ = Sfull.data.cpu().numpy()
        phase_ = phase.data.cpu().numpy()
        Sfull_spec = Sfull_ * np.exp(1.0j * phase_)
        S_sign = np.sign(np.real(Sfull_spec))
        S_sign = torch.from_numpy(S_sign).to(S.device)
        Sfull_spec_imag = np.imag(Sfull_spec)
        Sfull_spec_imag = torch.from_numpy(Sfull_spec_imag).unsqueeze(-1).to(S.device)
        Sfull = torch.mul(Sfull, S_sign).unsqueeze(-1)
        stft_matrix = concatenateFeature([Sfull, Sfull_spec_imag], dim=-1) # (B, F, T, 2)

        wav = istft_irfft(stft_matrix, hop_length=self.hop_size, win_length=self.nfft)
        return wav

    def forward(self, target_mag, target_phase, pred_mag, pred_phase):
        '''
        :param target_mag: (B, F-1, T)
        :param target_phase: (B, F, T)
        :param pred_mag: (B, F-1, T)
        :param pred_phase: (B, F, T)
        :return:
        '''
        target_wav = self.genWav(target_mag, target_phase)
        pred_wav = self.genWav(pred_mag, pred_phase)

        loss = self.loss(target_wav, pred_wav)
        return loss

############################# baseline UNet ############################# 
class UNet_block(nn.Module):
    def __init__(self, numIn, numOut, kernel_size, stride, leaky):
        super(UNet_block, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(numIn, numOut, kernel_size, stride, kernel_size // 2),
        nn.BatchNorm2d(numOut),
        nn.LeakyReLU(leaky)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x

class UNet_deblock(nn.Module):
    def __init__(self, numIn, numOut, kernel_size, stride, padding, output_padding, dropout=None, upsample=0):
        super(UNet_deblock, self).__init__()
        if upsample:
            self.deconv_block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(numIn, numOut, kernel_size, stride, kernel_size // 2),
                nn.BatchNorm2d(numOut),
                nn.ReLU()
            )
        else:
            if dropout != None:
                self.deconv_block = nn.Sequential(
                    nn.ConvTranspose2d(numIn, numOut, kernel_size, stride, padding, output_padding),
                    nn.BatchNorm2d(numOut),
                    nn.Dropout(dropout),
                    nn.ReLU()
                )
            else:
                self.deconv_block = nn.Sequential(
                    nn.ConvTranspose2d(numIn, numOut, kernel_size, stride, padding, output_padding),
                    nn.BatchNorm2d(numOut),
                    nn.ReLU()
                )


    def forward(self, x):
        x = self.deconv_block(x)
        return x

# Use as baseline. SINGING VOICE SEPARATION WITH DEEP
# U-NET CONVOLUTIONAL NETWORKS
# One less layer 
class UNet(nn.Module):
    def __init__(self, numIn=1, kernel_size=5):
        super(UNet, self).__init__()
        padding = kernel_size // 2
        self.conv1 = UNet_block(numIn, 16, kernel_size, 2, 0.2) # 1, 128,128 --> 16, 64, 64 c1
        self.conv2 = UNet_block(16, 32, kernel_size, 2, 0.2) # --> 32, 32, 32 c2
        self.conv3 = UNet_block(32, 64, kernel_size, 2, 0.2) # --> 64, 16, 16 c3
        self.conv4 = UNet_block(64, 128, kernel_size, 2, 0.2) # --> 128, 8, 8 c4
        self.conv5 = UNet_block(128, 256, kernel_size, 2, 0.2) # -->256, 4, 4 (bottom) c5
        self.deconv5 = UNet_deblock(256, 128, kernel_size, 2, padding, 1, 0.5) # --> 128, 8, 8 d5
        self.deconv4 = UNet_deblock(256, 64, kernel_size, 2, padding, 1, 0.5) # --> 64, 16, 16 d4
        self.deconv3 = UNet_deblock(128, 32, kernel_size, 2, padding, 1, 0.5) # --> 32, 32, 32
        self.deconv2 = UNet_deblock(64, 16, kernel_size, 2, padding, 1) # --> 16, 64, 64
        self.deconv1 = UNet_deblock(32, numIn, kernel_size, 2, padding, 1) # --> 1, 128, 128
        self.act = nn.Sigmoid() # new add
        self._initialize_weights()

    def forward(self, x):
        b,t,f = x.shape
        x = x.reshape(b, -1, t, f)
        x = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c5 = self.deconv5(c5)
        c5 = concatenateFeature([c5, c4], dim=1)
        c4 = self.deconv4(c5)
        c4 = concatenateFeature([c4, c3], dim=1)
        c3 = self.deconv3(c4)
        c3 = concatenateFeature([c3, c2], dim=1)
        c2 = self.deconv2(c3)
        c2 = concatenateFeature([c2, x], dim=1)
        x = self.deconv1(c2)
        x = self.act(x)
        x = x.reshape(b, t, f)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            init_weights(m)
