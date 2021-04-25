# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
import GenerateModels as models
from prepare_data_wsj import dataGenWSJ_fromMemory, postpro_and_gen
import torch.optim as optim
from tensorboardX import SummaryWriter
import sys
import warnings
import logging
import lera
import argparse
from utils import EarlyStopping, save_model, resume_model, load_model, Clock, \
    plot_and_compare, to_cuda, print_network
import soundfile as sf
from PIL import Image
import json
import shutil
import random
import utils

if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Restoration")

# model related
parser.add_argument('--model', default='EncoderNetLinear_lstm', type=str)
parser.add_argument('--loss', default='L1c', type=str, help='L1c, L1')

parser.add_argument('--newSeedNum', default=418571351248, type=int)  # new seeds for shuffle data when continue training
parser.add_argument('--epochs', default=30, type=int) # 30
parser.add_argument('--stepsPerEpochTrain', default=1100, type=int)  # when batchSize is 128, seq=128, default 1100
parser.add_argument('--stepsPerEpochValid', default=15, type=int) # default 15
parser.add_argument('--batchSize', default=128, type=int)  # when batchSize is 128, seq=128
parser.add_argument('--seqLen', default=128, type=int)
parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
parser.add_argument('--save_dir', default='/home/user/huangyating/HCRN/ckpt', type=str)
parser.add_argument('--title', default='HCRN2', type=str)
parser.add_argument('--checkpoint', type=str, default=None, help="checkpoint filename")
parser.add_argument('--useAct', type=str, default='Sigmoid', help="Use Activation function?")
parser.add_argument('--substract_mean', type=int, default=1, help='0 not substract mean')
parser.add_argument('--dBscale', type=int, default=1, help='0 not transfer to db scale')
parser.add_argument('--normalize', type=int, default=1, help='0 not normalize')
parser.add_argument('--add_blank', default=10, type=int, help="whether to add blank!")
parser.add_argument('--kernel_size', type=int, default=7, help='kernel size')
parser.add_argument('--dilation', type=int, default=1, help='dilation rate')

parser.add_argument('--lr', default=0.0002, type=float, help="Initial learning rate")
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=0.01, help='l2 loss')
parser.add_argument('--lr_decay', default=0, type=float, help="The value multiplied by lr at each epoch.")

parser.add_argument('--early_stop', type=bool, default=True, help='Early stop?')
parser.add_argument('--patience', type=int, default=10, help='patience') 
parser.add_argument('--validate_every', default=1, type=int, help="Validate every n epochs.")
parser.add_argument('--log_step_every', default=100, type=int, help="Validate every n epochs.")
parser.add_argument('--newOpt', default=0, type=int, help='Use a new optim?')

parser.add_argument('--cuda', default=True, type=bool, help="use cuda")
parser.add_argument('--ngpu', default=2, type=int, help="Number of gpu")
parser.add_argument('--tag', default='predWSJ', type=str, help="tag")
parser.add_argument('--leraT', default='PredWSJ', type=str, help='Title of the lera log')
parser.add_argument('--workers', default=4, type=int, help='Number of workers')
parser.add_argument('--logf', default='log_p1', type=str, help='filename of log file')

opt = parser.parse_args()
print(opt)

# config
hparams = utils.read_config('config.yaml')

stepsPerEpochTrain = opt.stepsPerEpochTrain
stepsPerEpochValid = opt.stepsPerEpochValid
# according to the size of training set and validation set 
# if opt.batchSize != 128 or opt.seqLen != 128:
#     ratio = 128 / opt.batchSize * 128 / opt.seqLen
#     stepsPerEpochTrain = int(opt.stepsPerEpochTrain * ratio)
#     stepsPerEpochValid = int(opt.stepsPerEpochValid * ratio)
# print('stepsPerEpochTrain {} stepsPerEpochValid {}'.format(stepsPerEpochTrain, stepsPerEpochValid))


def write_logs(str, scalar, steps):
    lera.log({str: scalar})
    print(str, scalar)
    writer.add_scalar(str, scalar, steps)

class Observer(object):
    def __init__(self, fn=None):
        if fn is not None:
            self.fn = fn
        else:
            self.fn = 'log'

        # read the previous log
        if os.path.exists(fn):
            with open(fn, 'r') as f:
                self.report_list = json.load(f)
        else:
            self.report_list = []

    def step(self, report_keys):
        report_dict_this_step = {}
        # print('Global vars', globals())
        for key in report_keys:
            if isinstance(globals()[key], torch.Tensor):
                report_dict_this_step[key] = globals()[key].item()
            else:
                report_dict_this_step[key] = globals()[key]
        self.report_list.append(report_dict_this_step)

        # write to the log file
        path = 'log.json'
        print(self.report_list)
        with open(path, 'w') as f:
            json.dump(self.report_list, f, indent=4)
        shutil.move(path, self.fn)

update_steps = 0
# stream logger
logger = logging.getLogger('mylogger')
fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(fomatter)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)
# file logger
observer = Observer(os.path.join(opt.save_dir + '_{}'.format(opt.title), opt.logf))

# save dir
if opt.save_dir is None:
    save_dir = './EncoderResult/Models/{}_{}'.format(opt.tag, opt.title)
else:
    save_dir = opt.save_dir + '_{}'.format(opt.title)
tmp_dir = os.path.join(save_dir, 'tmp')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

tmp_dir = os.path.join(save_dir, 'tmp')
tmp_train_dir = os.path.join(tmp_dir, 'train')
tmp_valid_dir = os.path.join(tmp_dir, 'valid')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
    os.makedirs(tmp_train_dir)
    os.makedirs(tmp_valid_dir)

logger.info('===> Setting seeds')
torch.manual_seed(opt.newSeedNum)

logger.info('===> Loading datasets')
data_generator = dataGenWSJ_fromMemory(opt.newSeedNum, opt.batchSize, opt.seqLen, opt.workers, opt.add_blank)
training_data_loader = data_generator.myDataGenerator(dataFlag=0, substract_mean=opt.substract_mean,
                                                      dBscale=opt.dBscale, normalize=opt.normalize)
validation_data_loader = data_generator.myDataGenerator(dataFlag=1, substract_mean=opt.substract_mean,
                                                        dBscale=opt.dBscale, normalize=opt.normalize)
# cuda and multi-gpu
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = torch.device("cuda" if opt.cuda else "cpu")

# model
logger.info('===> Building model and initializing')
if 'UNet' in opt.model:
    train_model = getattr(models, opt.model)(kernel_size=opt.kernel_size)
else:
    train_model = getattr(models, opt.model)(int(hparams.fft_size / 2),
                                                hparams.num_hidden_linear,
                                                num_frames=opt.seqLen, useAct=opt.useAct, kernel_size=opt.kernel_size, dilation_rate=opt.dilation)

if opt.loss == 'L1c':
    my_loss1 = nn.L1Loss()
    my_loss2 = getattr(models, 'WaveLoss')(dBscale=opt.dBscale, denormalize=opt.normalize,
                                            max_db=hparams.max_db, ref_db=hparams.ref_db,
                                            nfft=hparams.fft_size, hop_size=hparams.hop_size)
elif opt.loss == 'L1':
    my_loss = nn.L1Loss()
else:
    raise NotImplementedError

# optimizer
logger.info('===> Setting up optimizer')
optimizer = optim.Adam(train_model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
if opt.lr_decay == 0:
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
else:
    logger.info('===> Use step scheduler for SE')
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=opt.lr_decay)

# checkpoint
resume_epoch = 1
if opt.checkpoint is not None:
    logger.info('===> Loading a checkpoint for the main model')
    checkpoint_path = os.path.join(opt.save_dir + '_{}'.format(opt.title), opt.checkpoint)
    if opt.newOpt:
        logger.info('===> Use a new optimizer for main model')
        resume_epoch = load_model(train_model, checkpoint_path)
    else:
        logger.info('===> Continue training')
        resume_epoch = resume_model(train_model, optimizer, scheduler, checkpoint_path)

if opt.early_stop:
    early_stop = EarlyStopping(mode='min', patience=int(opt.patience / opt.validate_every))


if opt.ngpu > 1:
    train_model = nn.DataParallel(train_model, device_ids=list(range(opt.ngpu)))
train_model = train_model.to(device)
if opt.loss == 'L1c':
    my_loss1 = my_loss1.to(device)
    my_loss2 = my_loss2.to(device)
else:
    my_loss = my_loss.to(device)
to_cuda(optimizer, opt.cuda)
print_network(train_model)

tmp_train_dir = os.path.join(tmp_dir, 'train')
tmp_valid_dir = os.path.join(tmp_dir, 'valid')
if not os.path.exists(tmp_train_dir) or not os.path.exists(tmp_valid_dir):
    os.makedirs(tmp_train_dir)
    os.makedirs(tmp_valid_dir)

# print some info
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs_' + opt.tag + "_" + opt.title + '_' + opt.loss + '_' + str(resume_epoch) + "_" + current_time)
writer = SummaryWriter(log_dir)

# lera real-time log
lera.log_hyperparams({
    'title': opt.leraT + '_' + opt.title,
    'model': opt.model,
    'loss': opt.loss,
    'batchSize': opt.batchSize,
    'seqLen': opt.seqLen,
    'init_lr': opt.lr
})

def inspectAudio(input1, target, pred, BatchMixPhase, BatchTargetPhase, epoch, syn_num=2, train_or_valid='Train'):
    for i in range(syn_num):
        sample_input_linear = input1[i].data.cpu().numpy()  # (T, F-1)
        sample_target_linear = target[i].data.cpu().numpy()
        sample_pred_linear = pred[i].data.cpu().numpy()
        sample_phase_linear = BatchMixPhase[i]  # (T, F)
        sample_target_phase_linear = BatchTargetPhase[i]

        sample_input_wav = postpro_and_gen(sample_input_linear, sample_phase_linear, dBscale=opt.dBscale,
                                                 denormalize=opt.normalize)
        sample_target_wav = postpro_and_gen(sample_target_linear, sample_target_phase_linear, dBscale=opt.dBscale,
                                                  denormalize=opt.normalize)
        sample_pred_wav = postpro_and_gen(sample_pred_linear, sample_phase_linear, dBscale=opt.dBscale,
                                                denormalize=opt.normalize)

        if train_or_valid == 'Train':
            save_path = tmp_train_dir
        else:
            save_path = tmp_valid_dir

        # save the audios locally
        print('Saving to ', save_path)
        save_wav_name_prefix = 'epoch_{}_step_{}_samp_{}'.format(epoch, update_steps, i)
        sf.write(os.path.join(save_path, save_wav_name_prefix + '_input.wav'), sample_input_wav, hparams.sample_rate)
        sf.write(os.path.join(save_path, save_wav_name_prefix + '_target.wav'), sample_target_wav, hparams.sample_rate)
        sf.write(os.path.join(save_path, save_wav_name_prefix + '_pred.wav'), sample_pred_wav, hparams.sample_rate)
        print('Save sucessfully.')

        # give some plot!
        fig_name = save_wav_name_prefix + '_fig.jpg'
        fig_path = os.path.join(save_path, fig_name)
        plot_and_compare(sample_input_wav, sample_target_wav, sample_pred_wav, fig_path)
        I = Image.open(fig_path)

        # send to lera
        lera.log_audio('{} Random Sample {} input'.format(train_or_valid, i), sample_input_wav,
                       sample_rate=hparams.sample_rate)
        lera.log_audio('{} Random Sample {} target'.format(train_or_valid, i), sample_target_wav,
                       sample_rate=hparams.sample_rate)
        lera.log_audio('{} Random Sample {} pred'.format(train_or_valid, i), sample_pred_wav,
                       sample_rate=hparams.sample_rate)

        # send to lera to view real-time!
        lera.log_image('{} Random Sample {} specs'.format(train_or_valid, i), I)

# train an epoch
def train_an_epoch(epoch):
    train_model.train()

    total_train_loss = 0.0
    iter_num = 0
    global update_steps

    for _ in range(stepsPerEpochTrain):
        [BatchDataIn1, BatchDataOut, BatchMixPhase, BatchTargetPhase] = training_data_loader.__next__()
        iter_num += 1

        input1 = torch.from_numpy(BatchDataIn1)
        target = torch.from_numpy(BatchDataOut)

        # cuda
        if opt.cuda:
            input1 = input1.cuda()
            target = target.cuda()

        train_loss = 0

        train_model.zero_grad()

        T, F = input1[0].shape

        pred = train_model(input1)

        if opt.loss == 'L1c':
            train_loss_spec = my_loss1(target, pred) / (F * T) 
            # compute wave loss
            mix_phase = torch.from_numpy(BatchMixPhase)  # (B, T, F)
            target_phase = torch.from_numpy(BatchTargetPhase)
            if opt.cuda:
                mix_phase = mix_phase.cuda()
                target_phase = target_phase.cuda()
            # transpose
            target = torch.transpose(target, -2, -1)
            pred = torch.transpose(pred, -2, -1)
            target_phase = torch.transpose(target_phase, -2, -1)
            mix_phase = torch.transpose(mix_phase, -2, -1)
            train_loss_wav = my_loss2(target, target_phase, pred, mix_phase) / T

            train_loss = train_loss + train_loss_spec + train_loss_wav / 40

            lera.log({'train_loss_spec': train_loss_spec.item()})
            lera.log({'train_loss_wav': train_loss_wav.item()})

            print('train_loss_spec', train_loss_spec.item())
            print('train_loss_wav', train_loss_wav.item())

            writer.add_scalar('train_loss_spec', train_loss_spec.item(), update_steps)
            writer.add_scalar('train_loss_wav', train_loss_wav.item(), update_steps)
        else: 
            train_loss = train_loss + my_loss(target, pred) / (F * T)

        lera.log({'train_loss': train_loss.item()})
        print('train_loss', train_loss.item())
        writer.add_scalar('train_loss', train_loss.item(), update_steps)

        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()
        print('total_train_loss', total_train_loss)
        update_steps += 1
        print('Update_steps,', update_steps)

        if not update_steps % opt.log_step_every:
            # Spend some time to synthesize and save some samples
            print('Observing samples steps', update_steps)
            if opt.loss == 'L1c':
                target = torch.transpose(target, -2, -1)
                pred = torch.transpose(pred, -2, -1)
            inspectAudio(input1, target, pred, BatchMixPhase, BatchTargetPhase, epoch)

    global current_lr, avg_train_loss_this_epoch
    if iter_num != 0:
        avg_train_loss_this_epoch = total_train_loss / iter_num
        current_lr = scheduler.get_lr()[0]
        print('current lr', current_lr)
        lera.log({'avg_train_loss_per_epoch': avg_train_loss_this_epoch,
                  'lr': current_lr})
        writer.add_scalar('avg_train_loss_per_epoch', avg_train_loss_this_epoch, epoch)
        writer.add_scalar('lr', current_lr, epoch)

        # lr decay
        scheduler.step()
    else:
        print('Iter is 0, some error happened here')
        sys.exit(1)

def validate(epoch, best_valid_loss):
    train_model.eval()
    total_valid_loss = 0.0
    iter_num = 0
    begin_epoch = True

    with torch.no_grad():
        for _ in range(stepsPerEpochValid):
            [BatchDataIn1, BatchDataOut, BatchMixPhase, BatchTargetPhase] = validation_data_loader.__next__()

            iter_num += 1

            input1 = torch.from_numpy(BatchDataIn1)
            target = torch.from_numpy(BatchDataOut)

            # cuda
            if opt.cuda:
                input1 = input1.cuda()
                target = target.cuda()

            valid_loss = 0
            T, F = input1[0].shape

            pred = train_model(input1)

            if opt.loss == 'L1c':
                valid_loss_spec = my_loss1(target, pred) / (F * T)  # 除以这个系数其实是因为前面没有乘以权重，loss有点大

                # compute wave loss
                mix_phase = torch.from_numpy(BatchMixPhase)  # (B, T, F)
                target_phase = torch.from_numpy(BatchTargetPhase)
                if opt.cuda:
                    mix_phase = mix_phase.cuda()
                    target_phase = target_phase.cuda()
                # transpose
                target = torch.transpose(target, -2, -1)
                pred = torch.transpose(pred, -2, -1)
                target_phase = torch.transpose(target_phase, -2, -1)
                mix_phase = torch.transpose(mix_phase, -2, -1)

                valid_loss_wav = my_loss2(target, target_phase, pred, mix_phase) / T

                valid_loss = valid_loss + valid_loss_spec + valid_loss_wav / 40

                lera.log({'valid_loss_spec': valid_loss_spec.item()})
                lera.log({'valid_loss_wav': valid_loss_wav.item()})

                print('valid_loss_spec', valid_loss_spec.item())
                print('valid_loss_wav', valid_loss_wav.item())

                writer.add_scalar('valid_loss_spec', valid_loss_spec.item(), (epoch - 1)* stepsPerEpochValid + iter_num)
                writer.add_scalar('valid_loss_wav', valid_loss_wav.item(),  (epoch - 1)* stepsPerEpochValid + iter_num)
            else: 
                valid_loss = valid_loss + my_loss(target, pred) / (F * T)

            lera.log({'valid_loss': valid_loss.item()})
            print('valid_loss', valid_loss.item())
            writer.add_scalar('valid_loss', valid_loss.item(),  (epoch - 1)* stepsPerEpochValid + iter_num)
            total_valid_loss += valid_loss.item()

            # only log and plot in the begining when validation
            if begin_epoch:
                if opt.loss == 'L1c':
                    target = torch.transpose(target, -2, -1)
                    pred = torch.transpose(pred, -2, -1)
                inspectAudio(input1, target, pred, BatchMixPhase, BatchTargetPhase, epoch, train_or_valid='Valid')
            else:
                begin_epoch = False

        if iter_num != 0:
            avg_valid_loss_this_epoch = total_valid_loss / iter_num
            lera.log({'avg_valid_loss_per_epoch': avg_valid_loss_this_epoch})
            writer.add_scalar('avg_valid_loss_per_epoch', avg_valid_loss_this_epoch, epoch)

            ckpt_fn = 'ckpt_{}.pth'.format(epoch)
            # save a checkpoint
            save_model(train_model, optimizer, scheduler, epoch,
                       os.path.join(opt.save_dir + '_{}'.format(opt.title), ckpt_fn))

            if best_valid_loss is None or avg_valid_loss_this_epoch < best_valid_loss:
                best_valid_loss = avg_valid_loss_this_epoch
                ckpt_fn = 'ckpt_best.pth'
                # save the best
                save_model(train_model, optimizer, scheduler, epoch,
                           os.path.join(opt.save_dir + '_{}'.format(opt.title), ckpt_fn))

            return best_valid_loss, avg_valid_loss_this_epoch
        else:
            print('Iter is 0, some error happened here')
            sys.exit(1)

def main():
    best_valid_loss = None
    stopping = False
    clock = Clock()

    global update_epoch, avg_valid_loss_this_epoch, elapsed_time

    for epoch in range(resume_epoch, resume_epoch + opt.epochs):
        train_an_epoch(epoch)
        update_epoch = epoch  

        if not epoch % opt.validate_every:
            best_valid_loss, avg_valid_loss_this_epoch = validate(epoch, best_valid_loss)

            # True means early stop
            if avg_valid_loss_this_epoch:
                stopping = early_stop.step(avg_valid_loss_this_epoch)

            # write some log
            elapsed_time = clock.step()
            observer.step(['update_epoch', 'current_lr', 'avg_train_loss_this_epoch', 'avg_valid_loss_this_epoch',
                            'elapsed_time'])
        else:
            # write some log
            elapsed_time = clock.step()
            observer.step(['update_epoch', 'current_lr', 'avg_train_loss_this_epoch', 'elapsed_time'])

        if stopping:
            logger.info('===> Early stopping, the last epoch is {}'.format(epoch))
            break


    writer.close()


if __name__ == "__main__":
    main()
