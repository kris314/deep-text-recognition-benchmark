# Desc: orig(v0) + gt of target style image

import os
import sys
import time
import random
import string
import argparse
import glob
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import numpy as np
import torchvision.models as models
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.autograd as autograd
import pdb

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignPairCollate, Batch_Balanced_Dataset, tensor2im, save_image
from model import ModelV1, StyleTensorEncoder, StyleLatentEncoder, MsImageDisV2, AdaIN_Tensor_WordGenerator, VGGPerceptualLossModel, Mixer
from test_synth import validation_synth_v6

import tflib as lib
import tflib.plot

sys.path.append('/private/home/pkrishnan/codes/st-scribe/stylegan2-pytorch/')

try:
    import wandb

except ImportError:
    wandb = None

from model_word import ConditionalGenerator as styleGANGen 
from model_word import Discriminator as styleGANDis  
from non_leaking import augment
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def test(opt):
    lib.print_model_settings(locals().copy())

    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    log = open(os.path.join(opt.exp_dir,opt.exp_name,'log_dataset.txt'), 'a')
    AlignCollate_valid = AlignPairCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=False,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True, drop_last=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    if 'Attn' in opt.Prediction:
        converter = AttnLabelConverter(opt.character)
    else:
        converter = CTCLabelConverter(opt.character)
    
    
    opt.num_class = len(converter.character)

    g_ema = styleGANGen(opt.size, opt.latent, opt.n_mlp, opt.num_class, channel_multiplier=opt.channel_multiplier)
    g_ema = torch.nn.DataParallel(g_ema).to(device)
    g_ema.eval()

    print('model input parameters', opt.imgH, opt.imgW, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length)

    

    ## Loading pre-trained files
    if opt.modelFolderFlag:
        if len(glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth")))>0:
            opt.saved_synth_model = glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth"))[-1]

    if opt.saved_synth_model != '' and opt.saved_synth_model != 'None':
        print(f'loading pretrained synth model from {opt.saved_synth_model}')
        checkpoint = torch.load(opt.saved_synth_model)
        
        g_ema.load_state_dict(checkpoint['g_ema'], strict=False)
    
    # pdb.set_trace()
    if opt.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.module.mean_latent_content(opt.truncation_mean)
            
    else:
        mean_latent = None

    

    cntr=0

    
    for i,(image_input_tensors, image_gt_tensors, labels_1, labels_2) in  enumerate(valid_loader):
        print(i,len(valid_loader))
        image_input_tensors = image_input_tensors.to(device)
        image_gt_tensors = image_gt_tensors.to(device)
        batch_size = image_input_tensors.size(0)

        text_1, length_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_2, length_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        
        
        #forward pass from style and word generator
        if opt.fixedStyleBatch:
            fixstyle=[]
            # pdb.set_trace()
            style = mixing_noise(1, opt.latent, opt.mixing, device)
            fixstyle.append(style[0].repeat(opt.batch_size,1))
            if len(style)>1:
                fixstyle.append(style[1].repeat(opt.batch_size,1))
            style = fixstyle
        else:
            style = mixing_noise(opt.batch_size, opt.latent, opt.mixing, device)
        
        if 'CTC' in opt.Prediction:
            images_recon_2,_ = g_ema(style, text_2, input_is_latent=opt.input_latent, inject_index=5, truncation=opt.truncation, truncation_latent=mean_latent, randomize_noise=False)
        else:
            images_recon_2,_ = g_ema(style, text_2[:,1:-1], input_is_latent=opt.input_latent, inject_index=5, truncation=opt.truncation, truncation_latent=mean_latent, randomize_noise=False)
        
        # os.makedirs(os.path.join(opt.valDir,str(iteration)), exist_ok=True)
        for trImgCntr in range(batch_size):
            try:
                save_image(tensor2im(image_input_tensors[trImgCntr].detach()),os.path.join(opt.valDir,str(cntr)+'_'+str(trImgCntr)+'_sInput_'+labels_1[trImgCntr]+'.png'))
                save_image(tensor2im(image_gt_tensors[trImgCntr].detach()),os.path.join(opt.valDir,str(cntr)+'_'+str(trImgCntr)+'_csGT_'+labels_2[trImgCntr]+'.png'))
                save_image(tensor2im(images_recon_2[trImgCntr].detach()),os.path.join(opt.valDir,str(cntr)+'_'+str(trImgCntr)+'_csRecon_'+labels_2[trImgCntr]+'.png'))
            except:
                print('Warning while saving training image')
        cntr+=1

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp12/', help='Where to store logs and models')
    parser.add_argument('--exp_name', default='debug', help='Where to store logs and models')
    # parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=900000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_ocr_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_synth_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_gen_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--optim', default='adadelta', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--lr_policy', default='None', help='None|step')
    parser.add_argument('--step_size', type=int, default=100000, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='how much to decay learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1.0',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=24, help='maximum-label-length')
    parser.add_argument('--batch_min_length', type=int, default=1, help='minimum-label-length')
    parser.add_argument('--fixedString', action='store_true', help='use fixed length data')
    parser.add_argument('--batch_exact_length', type=int, default=4, help='exact-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=256, help='the width of the input image')
    # parser.add_argument('--ocr_imgH', type=int, default=32, help='the height of the input image')
    # parser.add_argument('--ocr_imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--pairText', action='store_true', help='use additional text for generation')
    parser.add_argument('--lexFile', default='/checkpoint/pkrishnan/datasets/vocab/english-words.txt', help='unqiue words in language')
    """ Model Architecture """
    # parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    # parser.add_argument('--FeatureExtraction', type=str, required=True,
    #                     help='FeatureExtraction stage. VGG|RCNN|ResNet')
    # parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    # parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--ocr_input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--char_embed_size', type=int, default=60, help='character embedding for content encoder')
    parser.add_argument('--ocrFixed', action='store_true', help='true: for pretrined OCR and fixed weights')
    parser.add_argument('--ocrWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--reconWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--disWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--vggPerWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--vggStyWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--styleDetach', action='store_true', help='whether to detach style')
    parser.add_argument('--gan_type', default='lsgan', help='lsgan/nsgan/wgan')
    parser.add_argument('--imgReconLoss', default='l1', help='l1/ssim/ms-ssim')
    parser.add_argument('--styleLoss', default='l1', help='l1/triplet')
    parser.add_argument('--contentLoss', default='pred', help='pred(ctc/attn)/vis/seq')
    parser.add_argument('--tripletMargin', type=float, default=1.0, help='triplet margin')
    parser.add_argument('--style_input', action='store_true', help='whether target style input is given for training/validation')
    parser.add_argument('--style_content_input', action='store_true', help='whether target input image is given for training/validation')

    parser.add_argument('--debugFlag', action='store_true', help='for debugging')
    parser.add_argument('--modelFolderFlag', action='store_true', help='load latest files from saved model folder')
    parser.add_argument('--testFlag', action='store_true', help='for testing')

    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--input_latent", action="store_true")
    parser.add_argument('--fixedStyleBatch', action='store_true', help='use fixed style in batch')
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)


    opt = parser.parse_args()
    
    if opt.rgb:
        opt.input_channel = 3

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name), exist_ok=True)
    # opt.trainDir=os.path.join(opt.exp_dir,opt.exp_name,'trainImages')
    opt.valDir=os.path.join(opt.exp_dir,opt.exp_name,'testImages'+'-tr'+str(opt.truncation))

    if opt.fixedStyleBatch:
        opt.valDir=opt.valDir+'-fsb'
    
    
    # opt.plotDir=os.path.join(opt.exp_dir,opt.exp_name,'plots')
    
    # os.makedirs(opt.trainDir, exist_ok=True)
    os.makedirs(opt.valDir, exist_ok=True)
    # os.makedirs(opt.plotDir, exist_ok=True)


    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    # random.seed(opt.manualSeed)
    # np.random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)
    # torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
