# -*- coding: utf-8 -*-
# Test and compute some evaluation metrics like pesq, stoi, sdr
import numpy as np
import os
import torch
import torch.nn as nn
import GenerateModels as models
from prepare_data_wsj import genWaveclip, postpro_and_gen, ExtractFeatureFromOneSignal_fromMemory, Add_holes_linear
from random import shuffle
import sys
import warnings
import logging
import argparse
from utils import load_model, Clock, plot_and_compare, to_cuda
import soundfile as sf
from separation import bss_eval_sources
from pypesq import pypesq
from pystoi.stoi import stoi
import random
from utils import SNR_db_to_scale
import pickle
import librosa
import utils


if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Restoration")

parser.add_argument('--model', default='EncoderNetLinear_lstm', type=str)
parser.add_argument('--loss', default='L1c', type=str)
parser.add_argument('--newSeedNum', default=123456789, type=int)  
parser.add_argument('--useAct', type=str, default='Sigmoid', help="Use Activation function?")
parser.add_argument('--seqLen', default=128, type=int)
parser.add_argument('--tag', type=str, default='best', help="Tag")
parser.add_argument('--checkpoint', type=str, default='ckpt_best.pth', help="checkpoint for the main model")
parser.add_argument('--postfix', type=str, default='blank10', help="Postfix of the result dir")
parser.add_argument('--save_dir', default='/home/user/huangyating/HCRN/ckpt')
parser.add_argument('--title', default='HCRN', type=str)
parser.add_argument('--substract_mean', type=int, default=1, help='0 not substract mean')
parser.add_argument('--cuda', default=True, type=bool, help="use cuda")
parser.add_argument('--repeat', default=1, type=int, help="repeat n times")
parser.add_argument('--add_blank', default=10, type=int, help="whether to add blank!") # We can specify the consecutive blank frames here
parser.add_argument('--valid_or_test', default='test', type=str, help="Valid or test!")
parser.add_argument('--dBscale', type=int, default=1, help='0 not transfer to db scale')
parser.add_argument('--normalize', type=int, default=1, help='0 not normalize')
parser.add_argument('--clip_phase', default=1, type=int, help="1 clip phase")
parser.add_argument('--kernel_size', type=int, default=7, help='kernel size')
parser.add_argument('--dilation', type=int, default=1, help='dilation rate')
parser.add_argument('--db', type=float, default=5, help='mixing db') # We can specify the mixing dB level here
parser.add_argument('--compute_orig', type=int, default=1, help='compute original metrics')

opt = parser.parse_args()
print(opt)

# config
hparams = utils.read_config('config.yaml')

random.seed(opt.newSeedNum)
update_steps = 0
# stream logger
logger = logging.getLogger('mylogger')
fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(fomatter)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)

logger.info('===> Setting seeds')
torch.manual_seed(opt.newSeedNum)

# model
logger.info('===> Building a model')
if 'UNet' in opt.model:
    train_model = getattr(models, opt.model)()
else:
    train_model = getattr(models, opt.model)(int(hparams.fft_size / 2),
                                            hparams.num_hidden_linear,
                                            num_frames=opt.seqLen, useAct=opt.useAct, kernel_size=opt.kernel_size, dilation_rate=opt.dilation)


resume_epoch = 1
logger.info('===> Loading the model')
checkpoint_path = os.path.join(opt.save_dir + '_{}'.format(opt.title), opt.checkpoint)
resume_epoch = load_model(train_model, checkpoint_path)
print('Loading model from epoch {}'.format(resume_epoch))

# cuda and multi-gpu
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = torch.device("cuda" if opt.cuda else "cpu")
train_model = train_model.to(device)

print(train_model)

target_file = hparams.target_fn_test
interf_file = hparams.interf_fn_test

def load_pkl(fn):
    with open(fn, 'rb') as f:
        test_dict = pickle.load(f)
    test_list = list(test_dict.keys())
    return test_dict, test_list

target_dict, target_list = load_pkl(target_file)
interf_dict, interf_list = load_pkl(interf_file)
interf_audio_list = list(filter(lambda s: 'audioset' in s, interf_list))
interf_esc_list = list(filter(lambda s: 'ESC' in s, interf_list))

halfFFT = int(hparams.fft_size / 2)
if opt.add_blank:
    add_hole = Add_holes_linear(opt.seqLen, opt.add_blank, halfFFT, opt.newSeedNum)
else:
    add_hole = None

# save dir
save_dir = opt.save_dir + '_{}'.format(opt.title)
result_dir = os.path.join(save_dir, 'result')
wav_dir = os.path.join(result_dir, '{}_wav_{}_{}'.format(opt.valid_or_test, opt.tag, opt.postfix))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(wav_dir):
    os.makedirs(wav_dir)

def infer_a_sample(sample_i, round):
    target_path = target_list[sample_i]
    mix_flag = random.random()
    if mix_flag < 0.33:
        interf_id = random.randint(0, len(target_list) - 1)
        interf_path = target_list[interf_id]
        interf = target_dict[interf_path]
    elif 0.33 <= mix_flag < 0.66:
        interf_id = random.randint(0, len(interf_audio_list) - 1)
        interf_path = interf_audio_list[interf_id]
        interf = interf_dict[interf_path]
    else:
        interf_id = random.randint(0, len(interf_esc_list) - 1)
        interf_path = interf_esc_list[interf_id]
        interf = interf_dict[interf_path]
    target = target_dict[target_path]
    chooseIndexNormalised = None
    batchLen = opt.seqLen
    saveMixtureName = None
    interf_scale = SNR_db_to_scale(opt.db)
    print('Mixing scale is {}'.format(interf_scale))
    (batchInput1, batchTarget, batchMixphase, batchTargetphase, chooseIndex) = ExtractFeatureFromOneSignal_fromMemory(target, interf, interf_scale,
                chooseIndexNormalised, batchLen, add_hole, saveMixtureName, opt.dBscale, hparams.sample_rate,
                opt.normalize)

    phase_dim = halfFFT + 1
    L = chooseIndex[-1] + opt.seqLen

    train_model.eval()

    with torch.no_grad():
        batchInput1 = torch.from_numpy(batchInput1)

        if opt.cuda:
            batchInput1 = batchInput1.cuda()

        pred = train_model(batchInput1)

    pred_spec = np.zeros((L, halfFFT))
    gt_spec = np.zeros((L, halfFFT))
    mix_spec = np.zeros((L, halfFFT))
    mix_phase = np.zeros((L, phase_dim))
    target_phase = np.zeros((L, phase_dim))

    try: # in case the wav is shorter than seqLen frames
        for n, i in enumerate(chooseIndex):
            # the mixture
            mix_spec[i: i + opt.seqLen] = batchInput1[n].data.cpu().numpy()
            # the gt
            gt_spec[i: i + opt.seqLen] = batchTarget[n]
            # the prediction
            pred_spec[i: i + opt.seqLen] = pred[n].data.cpu().numpy()

            mix_phase[i: i + opt.seqLen] = batchMixphase[n]
            target_phase[i: i + opt.seqLen] = batchTargetphase[n]

        # Forgot to add options!!
        mix_wav = postpro_and_gen(mix_spec, mix_phase, dBscale=opt.dBscale, denormalize=opt.normalize)
        target_wav = postpro_and_gen(gt_spec, target_phase, dBscale=opt.dBscale, denormalize=opt.normalize)
        pred_wav = postpro_and_gen(pred_spec, mix_phase, dBscale=opt.dBscale, denormalize=opt.normalize)

        mixtureSaveName = 'r_{}_samp_{}_input.wav'.format(round, sample_i)
        targetSaveName = 'r_{}_samp_{}_target.wav'.format(round, sample_i)
        predSaveName = 'r_{}_samp_{}_pred.wav'.format(round, sample_i)
        figName = 'r_{}_samp_{}_fig.jpg'.format(round, sample_i)

        # save the audios locally
        sf.write(os.path.join(wav_dir, mixtureSaveName), mix_wav, hparams.sample_rate)
        sf.write(os.path.join(wav_dir, targetSaveName), target_wav, hparams.sample_rate)
        sf.write(os.path.join(wav_dir, predSaveName), pred_wav, hparams.sample_rate)
        print('Saving to r_{}_samp_{}'.format(round, sample_i))

        # compute scores
        sdr_this_sample = bss_eval_sources(target_wav, pred_wav)[0]
        pesq_this_sample = pypesq(hparams.sample_rate, target_wav, pred_wav, 'nb')
        stoi_this_sample = stoi(target_wav, pred_wav, hparams.sample_rate, extended=False)

        # plot fig
        plot_and_compare(mix_wav, target_wav, pred_wav, os.path.join(wav_dir, figName), hparams.sample_rate)

        print('SDR, PESQ, STOI this sample is {}, {}, {}.'.format(sdr_this_sample, pesq_this_sample, stoi_this_sample))

        if opt.compute_orig:
            sdr_orig_this_sample = bss_eval_sources(target_wav, mix_wav)[0]
            pesq_orig_this_sample = pypesq(hparams.sample_rate, target_wav, mix_wav, 'nb')
            stoi_orig_this_sample = stoi(target_wav, mix_wav, hparams.sample_rate, extended=False)
            print('SDR, PESQ, STOI this sample is {}, {}, {} orig.'.format(sdr_orig_this_sample, pesq_orig_this_sample,
                                                                      stoi_orig_this_sample))

    except Exception as e:
        sdr_this_sample = None
        pesq_this_sample = None
        stoi_this_sample = None
        sdr_orig_this_sample = None
        pesq_orig_this_sample = None
        stoi_orig_this_sample = None
        print(e)

    if opt.compute_orig:
        return sdr_this_sample, pesq_this_sample, stoi_this_sample, \
               sdr_orig_this_sample, pesq_orig_this_sample, stoi_orig_this_sample
    else:
        return sdr_this_sample, pesq_this_sample, stoi_this_sample

def main():
    num_samples = len(target_list)

    print('Repeat {} times, totally {} samples.'.format(opt.repeat, num_samples * opt.repeat))

    sdr_list = []
    pesq_list = []
    stoi_list = []

    if opt.compute_orig:
        sdr_orig_list = []
        pesq_orig_list = []
        stoi_orig_list = []

    for r in range(opt.repeat):
        test_i = 0
        for i in range(num_samples):
            if opt.compute_orig:
                sdr_i, pesq_i, stoi_i, sdr_o_i, pesq_o_i, stoi_o_i = infer_a_sample(test_i, r)
            else:
                sdr_i, pesq_i, stoi_i  = infer_a_sample(test_i, r)

            if sdr_i is not None:
                sdr_list.append(sdr_i)
                pesq_list.append(pesq_i)
                stoi_list.append(stoi_i)
                if opt.compute_orig:
                    sdr_orig_list.append(sdr_o_i)
                    pesq_orig_list.append(pesq_o_i)
                    stoi_orig_list.append(stoi_o_i)

            test_i += 1

        shuffle(interf_list)

    mean_sdr = np.mean(sdr_list)
    mean_pesq = np.mean(pesq_list)
    mean_stoi = np.mean(stoi_list)
    print('Mean SDR, PESQ, STOI is {}, {}, {}'.format(mean_sdr, mean_pesq, mean_stoi))

    if opt.compute_orig:
        mean_sdr_o = np.mean(sdr_orig_list)
        mean_pesq_o = np.mean(pesq_orig_list)
        mean_stoi_o = np.mean(stoi_orig_list)
        print('Mean SDR, PESQ, STOI is {}, {}, {} orig'.format(mean_sdr_o, mean_pesq_o, mean_stoi_o))

if __name__ == '__main__':
    main()
