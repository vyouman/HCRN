# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import pickle
import os
import soundfile as sf
import resampy
import numpy as np
import random
import librosa
import utils
from random import shuffle
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import datetime
from utils import SNR_db_to_scale
import librosa

# train_set = '/home/user/huangyating/espnet/egs/wsj/tts1/data/train_si284/wav.scp'
# valid_set = '/home/user/huangyating/espnet/egs/wsj/tts1/data/test_dev93/wav.scp'
# test_set = '/home/user/huangyating/espnet/egs/wsj/tts1/data/test_eval92/wav.scp'
# format: uttid  wav_path

hparams = utils.read_config('config.yaml')

def complex2float(array):
    shape=array.shape
    o=array.real.reshape(1, shape[0],shape[1]).repeat(2,axis=0)
    o[1]=array.imag
    return o

def float2complex(real_and_img): # (2,F,T)
    real = real_and_img[0]
    img = real_and_img[1]
    return real + 1j * img 

def postpro_and_gen(S, phase, returnS = 0, dBscale = 1, denormalize=1, complex_phase=0, clip_phase=0): # T, F
    if dBscale:
        if denormalize:
            # denormalization
            S = S * hparams.max_db - hparams.max_db + hparams.ref_db
        S = librosa.db_to_amplitude(S)

    # pad with 0
    Sfull = np.concatenate((S, np.zeros(shape=(S.shape[0], 1))), axis=-1)

    if clip_phase:
        phase = np.concatenate((phase, np.zeros(shape=(2, phase.shape[1], 1))), axis=-1)

    # generate waveform
    wav = genWaveclip(Sfull, phase, complex_phase)
    if not returnS:
        return wav
    else:
        return wav, Sfull, phase

def postpro_and_gen_noPad(S, phase, returnS = False, dBscale = 1, denormalize=1): # T, F
    if dBscale:
        if denormalize:
            # denormalization
            S = S * hparams.max_db - hparams.max_db + hparams.ref_db
        S = librosa.db_to_amplitude(S)

    # generate waveform
    wav = genWaveclip(S, phase)
    if not returnS:
        return wav
    else:
        return wav, S

def genWaveclip(mag, phase, complex_phase=0):
    if complex_phase:
        phase = float2complex(phase) # (T, F)
        spec = mag * phase
    else:
        spec = mag * np.exp(1.0j * phase)
    spec = spec.transpose()
    wav = librosa.core.spectrum.istft(spec, hparams.hop_size)
    return wav

def linearspectrogram(y, dBscale = 1, normalize=1, complex_phase=0):
    D = librosa.core.spectrum.stft(y, hparams.fft_size, hparams.hop_size) # F, T
    if complex_phase:
        S, phase = librosa.magphase(D)
        phase = complex2float(phase)
        # print('Phase dim in linearspec', phase.shape)
    else:
        S = np.abs(D)
    if dBscale:
        S = librosa.amplitude_to_db(S)
        if normalize:
            # normalization
            S = np.clip((S - hparams.ref_db + hparams.max_db) / hparams.max_db, 1e-8, 1)
    if complex_phase:
        return S, phase
    else:
        return S, np.angle(D)

def mix_target_noise(s1, s2, s2_scale):
    L = len(s1)
    s2_new = s2.copy()
    while len(s2) < L:
        s2 = np.concatenate((s2, s2_new), axis=0)
    L2 = len(s2)

    if L2 > L:
        start = random.randint(0, L2 - L)
        s2 = s2[start: start + L]

    if (s1 is None) | (s2 is None):
        print("Data loading fail")
        sys.exit()

    s2 = s2 * s2_scale
    mixture = s1 + s2
    if hparams.rescaling:
        scale = 1 / max(abs(mixture)) * hparams.rescaling_max
    else:
        scale = 1 / max(abs(mixture)) * 0.99  # normalise the mixture thus the maximum magnitude = 0.99
    mixture *= scale
    target = s1 * scale
    interf = s2 * scale
    return mixture, target, interf

# Dataset for the train set, generate dataset on the fly. only for model validation......
class Add_holes_linear(object):
    def __init__(self, seqLen= 128, masked_size= 10, linear_dim = 129,  seedNum=None, boundary_margin = 11):
        super(Add_holes_linear, self).__init__()

        if seedNum is not None:
            random.seed(seedNum)

        self.patchSize = seqLen
        self.masked_size = masked_size
        if boundary_margin is None:
            self.boundary_margin = masked_size
        else:
            self.boundary_margin = boundary_margin

    def add_blank(self, input_linear):
        start_t = random.randrange(self.boundary_margin,(self.patchSize - self.masked_size) - self.boundary_margin)
        input_linear[start_t: start_t + self.masked_size, :] = 0

def ExtractFeatureFromOneSignal_fromMemory(s1, s2, s2_scale, chooseIndexNormalised, batchLen,
                                      add_hole, saveNameMix=None, dBscale = 0, sr=8000, normalize=1, complex_phase=0, phase_add_blank=0, clip_phase=0):
    '''
    note that s1 and s2 are already normalized
    '''

    # MODIFIED BY @vyouman
    mixture, target, _ = mix_target_noise(s1, s2, s2_scale)

    # save
    if saveNameMix is not None:
        librosa.output.write_wav(saveNameMix, mixture, sr=sr)

    try:
        # if complex_phase:
        #     print('Use complex phase!')
        (mixLogPower, mixPhase) = linearspectrogram(mixture, dBscale, normalize, complex_phase)
        (targetLogPower, targetPhase) = linearspectrogram(target, dBscale, normalize, complex_phase)
        mixLogPower = mixLogPower.astype(np.float32) # (F, T)
        targetLogPower = targetLogPower.astype(np.float32)
        mixPhase = mixPhase.astype(np.float32)
        targetPhase = targetPhase.astype(np.float32)
        mixLogPower = mixLogPower[:-1]
        targetLogPower = targetLogPower[:-1]
        # print('Mixphase dim in extract', mixPhase.shape)
        if clip_phase:
            mixPhase = mixPhase[:, :-1]
            targetPhase = targetPhase[:, :-1]
            # print('Mixphase dim in extract after clip', mixPhase.shape) # 2, F, T

        # If you want to repeat your results, use this one
        if chooseIndexNormalised is None:
            chooseIndex = np.arange(0, mixLogPower.shape[1], batchLen, dtype=int)
            chooseIndex[-1] = mixLogPower.shape[1] - batchLen
        else:
            chooseIndex = (chooseIndexNormalised * (mixLogPower.shape[1] - batchLen)).astype(int)

        N = len(chooseIndex)

        # concatenate the feature as the input
        Index1 = (np.tile(range(0, batchLen), (N, 1))).T
        Index2 = np.tile(chooseIndex, (batchLen, 1))
        Index = Index1 + Index2  # (batchLen, N)

        # DNN input
        mixLogPower = mixLogPower[:, Index]  # (W,T)--->(W,100,N)
        subBatchIn1 = mixLogPower.transpose([2, 1, 0]) # (N, n_frames, W)
        if complex_phase:
            mixPhase = mixPhase[:, :, Index] # (2, W, T) --> (2, W, 100, N)
            mixPhase = mixPhase.transpose([3, 0, 2, 1]) # (2, W, 100, N) --> (N, 2, seqLen, W+1)
            targetPhase = targetPhase[:, :, Index]
            targetPhase = targetPhase.transpose([3, 0, 2, 1])
            # print('Mixphase dim in extract after transpose', mixPhase.shape)
        else:
            mixPhase = mixPhase[:, Index]
            mixPhase = mixPhase.transpose([2, 1, 0]) # (N, seqLen, W+1) or (N, seqLen, W+1, 2)
            targetPhase = targetPhase[:, Index]
            targetPhase = targetPhase.transpose([2, 1, 0])

        if add_hole is not None:
            # ADD BY @vyouman, ADD HOLES!!
            nclips, _, _ = subBatchIn1.shape
            # print('Nclips', nclips)
            for j in range(nclips):
                if phase_add_blank:
                    add_hole.add_blank(subBatchIn1[j], mixPhase[j])
                else:
                    add_hole.add_blank(subBatchIn1[j])

        # DNN output
        subBatchOut = targetLogPower[:, Index]  # (W,T)--->(W,100,N) the target linear spectrum
        subBatchOut = subBatchOut.transpose([2, 1, 0])  # (W,100,N, 3)---> (N,100,W)

        currentSubBatchIn1 = subBatchIn1[:N]
        currentSubBatchOut = subBatchOut[:N]
        currentSubMixPhase = mixPhase[:N]
        currentSubTargetPhase = targetPhase[:N]
    except Exception as e:
        print('Cracks in extracting the features, may be linearspectrogram')
        print(e)
        currentSubBatchIn1 = None
        currentSubBatchOut = None
        currentSubMixPhase = None
        currentSubTargetPhase = None
        chooseIndex = None

    return (currentSubBatchIn1, currentSubBatchOut, currentSubMixPhase, currentSubTargetPhase, chooseIndex)

class dataGenWSJ_fromMemory(object):

    def __init__(self, seedNum = 123456789, batch_size = 256, seqLen = 128, workers = None, add_blank=10, verbose = True, verboseDebugTime = True, complex_phase=0, phase_add_blank=0, clip_phase=0):
        self.seedNum = seedNum
        self.verbose = verbose
        self.verboseDebugTime = verboseDebugTime

        # num_workers = min(cpu_count()-2,6) # parallel at most 4 threads
        if workers is not None:
            num_workers = workers
        else:
            num_workers = cpu_count() - 2
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.parallelN = num_workers

        self.BATCH_SIZE_Train = batch_size 

        self.BATCH_SIZE_Valid = batch_size 

        self.batchLen = seqLen  # 2^N for ease of model design

        self.halfNFFT = int(hparams.fft_size/2)  # instead of nfft/2+1, we keep only 2^N for ease of model design

        self.target_train_i = 0
        self.target_valid_i = 0
        self.target_test_i = 0

        target_fn = hparams.target_fn
        interf_fn = hparams.interf_fn

        def load_pkl(fn):
            with open(fn, 'rb') as f:
                [tr, cv]=pickle.load(f)
            tr_list = list(tr.keys())
            cv_list = list(cv.keys())
            return tr, cv, tr_list, cv_list

        self.target_train_items, self.target_valid_items, self.target_train, self.target_valid = load_pkl(target_fn)
        self.interf_train_items, self.interf_valid_items, self.interf_train, self.interf_valid = load_pkl(interf_fn)
        self.trainNum = len(self.target_train)
        self.validNum = len(self.target_valid)
        self.train_i = 0
        self.valid_i = 0

        # Repeatable!
        random.seed(seedNum)
        shuffle(self.target_train)
        shuffle(self.target_valid)
        shuffle(self.interf_train)
        shuffle(self.interf_valid)

        self.complex_phase = complex_phase
        self.phase_add_blank = phase_add_blank
        self.clip_phase = clip_phase

        if self.phase_add_blank or self.complex_phase:
            print('Not implemented')
            exit()
        else:
            self.add_hole = Add_holes_linear(self.batchLen, add_blank, self.halfNFFT, seedNum)

    def myDataGenerator(self, dataFlag=0, substract_mean=1, dBscale=0, normalize=1):
        batchSize = [self.BATCH_SIZE_Train, self.BATCH_SIZE_Valid][dataFlag]

        BatchDataIn1 = np.zeros((batchSize, self.batchLen, self.halfNFFT), dtype='f')  # 100 x 512  linear-scale
        # size (number_sample, self.batchLen, #Mels)  100 x Wm
        BatchDataOut = np.zeros((batchSize, self.batchLen, self.halfNFFT), dtype='f')

        # save for the next batch
        BatchDataIn1Next = np.zeros((batchSize * 2, self.batchLen, self.halfNFFT), dtype='f')
        BatchDataOutNext = np.zeros((batchSize * 2, self.batchLen, self.halfNFFT), dtype='f')

        # ADD BY @vyouman
        phase_f_dim = self.halfNFFT + 1 if not self.clip_phase else self.halfNFFT   
        BatchMixPhase = np.zeros((batchSize, self.batchLen, phase_f_dim), dtype='f')
        BatchTargetPhase = np.zeros((batchSize, self.batchLen, phase_f_dim), dtype='f')
        BatchMixPhaseNext = np.zeros((batchSize * 2, self.batchLen, phase_f_dim), dtype='f')
        BatchTargetPhaseNext = np.zeros((batchSize * 2, self.batchLen, phase_f_dim), dtype='f')

        batchNum = 0
        availableN = 0 # number of unused samples generated from the previous round of parallel executor

        if substract_mean:
            print("Subtracting mean while normalising s1 and s2")

        target_item_dict = [self.target_train_items, self.target_valid_items][dataFlag]
        interf_item_dict = [self.interf_train_items, self.interf_valid_items][dataFlag]
        target_path_list = [self.target_train, self.target_valid][dataFlag]
        interf_path_list = [self.interf_train, self.interf_valid][dataFlag]

        interf_audio_list = list(filter(lambda s: 'audioset' in s, interf_path_list))
        interf_esc_list = list(filter(lambda s: 'ESC' in s, interf_path_list))
        print('Interf audioset num {}'.format(len(interf_audio_list)))
        print('Interf ESC num {}'.format(len(interf_esc_list)))

        while 1:
            if self.verbose:
                print('\nNow collect a mini batch for {}'.format(['training','validataion'][dataFlag]))

            time_collect_start = datetime.datetime.now()
            NinCurrentBatch=0

            if availableN>0:
                tempAvailableN = min(availableN, batchSize)
                # print('\n Grab unused {} samples from the previous round of parallel processing'.format(tempAvailableN))
                BatchDataIn1[:tempAvailableN] = BatchDataIn1Next[:tempAvailableN]
                BatchDataOut[:tempAvailableN] = BatchDataOutNext[:tempAvailableN]
                BatchMixPhase[:tempAvailableN] = BatchMixPhaseNext[:tempAvailableN]
                BatchTargetPhase[:tempAvailableN] = BatchTargetPhaseNext[:tempAvailableN]
                availableN = max(availableN-tempAvailableN,0)
                # There are too many unused samples from the previous round
                if availableN>0:
                    BatchDataIn1Next[:availableN] = BatchDataIn1Next[tempAvailableN:tempAvailableN+availableN]
                    BatchDataOutNext[:availableN] = BatchDataOutNext[tempAvailableN:tempAvailableN + availableN]
                    BatchMixPhaseNext[:availableN] = BatchMixPhaseNext[tempAvailableN:tempAvailableN + availableN]
                    BatchTargetPhaseNext[:availableN] = BatchTargetPhaseNext[tempAvailableN:tempAvailableN + availableN]
                NinCurrentBatch += tempAvailableN

            while NinCurrentBatch<batchSize:
                futures = []

                # for each target sequence, randomly choose an interfering environmental signal and add them together

                sequence_i = [self.train_i,self.valid_i][dataFlag]

                for sequence_ii in range(sequence_i, sequence_i + self.parallelN):  # parallel 4 processes
                    # futures.append(self.executor.submit(partial(foo, dataFlag, sequence_ii)))

                    target_path = target_path_list[sequence_ii]
                    mix_flag = random.random()
                    if mix_flag < 0.33:
                        interf_id = random.randint(0, len(target_path_list) - 1)
                        interf_path = target_path_list[interf_id]
                        interf = target_item_dict[interf_path]
                        dB = 15 * random.random()
                    elif 0.33 <= mix_flag < 0.66:
                        interf_id = random.randint(0, len(interf_audio_list) - 1)
                        interf_path = interf_audio_list[interf_id]
                        interf = interf_item_dict[interf_path]
                        dB = 15 * random.random()
                    else:
                        interf_id = random.randint(0, len(interf_esc_list) - 1)
                        interf_path = interf_esc_list[interf_id]
                        interf = interf_item_dict[interf_path]
                        dB = 15 * random.random()

                    # print('\n ========={}======{}======={}========{}=======\n'.format(self.train_i, sequence_ii, target_path, interf_path))

                    target = target_item_dict[target_path]

                    chooseIndexNormalised = None

                    batchLen = self.batchLen

                    saveMixtureName = None
                    interf_scale = SNR_db_to_scale(dB)
                    print('Mixing scale is {}'.format(interf_scale))
                    futures.append(self.executor.submit(
                        partial(ExtractFeatureFromOneSignal_fromMemory, target, interf, interf_scale,
                                chooseIndexNormalised, batchLen, self.add_hole, saveMixtureName, dBscale, hparams.sample_rate, normalize, self.complex_phase, self.phase_add_blank, self.clip_phase)))

                # [print(future.result()[0][0, 40, 30, 0]) for future in futures]
                tempResults = [future.result() for future in futures]

                # (currentSubBatchIn, currentSubBatchOut) = self.ExtractFeatureAssociatedOneTarget(dataFlag, sequence_i)

                for (currentSubBatchIn1, currentSubBatchOut, currentSubMixPhase,  currentSubTargetPhase, _) in tempResults:

                    if currentSubBatchIn1 is not None:
                        N = len(currentSubBatchIn1)
                        if NinCurrentBatch + N > batchSize:
                            # these samples are not used in the current batch, we will save it for the next batch of data generation
                            N = batchSize - NinCurrentBatch
                            reuseableN = min(len(currentSubBatchIn1)+NinCurrentBatch-batchSize,batchSize*2-availableN)
                            BatchDataIn1Next[availableN:availableN + reuseableN] = currentSubBatchIn1[N:N+reuseableN]
                            BatchDataOutNext[availableN:availableN + reuseableN] = currentSubBatchOut[N:N+reuseableN]
                            BatchMixPhaseNext[availableN:availableN + reuseableN] = currentSubMixPhase[N:N+reuseableN]
                            BatchTargetPhaseNext[availableN:availableN + reuseableN] = currentSubTargetPhase[N:N+reuseableN]
                            availableN += reuseableN
                        if N>0:
                            BatchDataIn1[NinCurrentBatch:NinCurrentBatch + N] = currentSubBatchIn1[:N]
                            BatchDataOut[NinCurrentBatch:NinCurrentBatch + N] = currentSubBatchOut[:N]
                            BatchMixPhase[NinCurrentBatch:NinCurrentBatch + N] = currentSubMixPhase[:N]
                            BatchTargetPhase[NinCurrentBatch:NinCurrentBatch + N] = currentSubTargetPhase[:N]
                            NinCurrentBatch += N
                    else:
                        print('CurrentSubBatch is None, some errors happened')

                if dataFlag == 0:
                    self.train_i += self.parallelN
                    if (self.train_i >= self.trainNum - self.parallelN + 1):
                        self.train_i = 0
                        shuffle(self.target_train)
                elif dataFlag == 1:
                    self.valid_i += self.parallelN
                    if (self.valid_i >=self.validNum-self.parallelN + 1):
                        self.valid_i = 0
                        shuffle(self.target_valid)



            time_collect_end = datetime.datetime.now()
            print("\t The total time to collect the current batch of data is ", time_collect_end - time_collect_start)

            batchNum += 1

            if self.verbose:
                print('\n Batch {} data collected using time of '.format(batchNum), time_collect_end - time_collect_start, '\n')

            yield [BatchDataIn1, BatchDataOut, BatchMixPhase, BatchTargetPhase]
