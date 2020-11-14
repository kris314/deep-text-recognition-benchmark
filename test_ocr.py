# Desc: orig(v0) + gt of target style image

import os
import sys
import time
import random
import string
import argparse
import glob
import math
import re

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
import html_visual as html
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity
from nltk.metrics.distance import edit_distance
from skimage.transform import resize
from skimage.io import imsave
import skimage
import cv2
import tflib as lib

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, tensor2im, save_image, phoc_gen, text_gen, text_gen_synth, LmdbDataset
from model import ModelV1, GlobalContentEncoder, VGGPerceptualEmbedLossModel, VGGFontModel
from modules.feature_extraction import ResNet_StyleExtractor, VGG_ContentExtractor, ResNet_StyleExtractor_WIN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)

def test(opt):
    lib.print_model_settings(locals().copy())

    if 'Attn' in opt.Prediction:
        converter = AttnLabelConverter(opt.character)
        text_len = opt.batch_max_length+2
    else:
        converter = CTCLabelConverter(opt.character)
        text_len = opt.batch_max_length

    opt.classes = converter.character
    
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    
    valid_dataset = LmdbDataset(root=opt.test_data, opt=opt)
    test_data_sampler = data_sampler(valid_dataset, shuffle=False, distributed=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, 
        shuffle=False,  # 'True' to check training progress with validation function.
        sampler=test_data_sampler,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True, drop_last=False)
    
    print('-' * 80)

    opt.num_class = len(converter.character)
    
    ocrModel = ModelV1(opt).to(device)

    ## Loading pre-trained files
    print(f'loading pretrained ocr model from {opt.saved_ocr_model}')
    checkpoint = torch.load(opt.saved_ocr_model, map_location=lambda storage, loc: storage)
    ocrModel.load_state_dict(checkpoint)
    
    evalCntr=0
    fCntr=0
    
    c1_s1_input_correct=0.0
    c1_s1_input_ed_correct=0.0
    # pdb.set_trace()
        
    for vCntr, (image_input_tensors, labels_gt) in enumerate(valid_loader):
        print(vCntr)
        
        image_input_tensors = image_input_tensors.to(device)
        text_gt, length_gt = converter.encode(labels_gt, batch_max_length=opt.batch_max_length)

        with torch.no_grad():
            currBatchSize = image_input_tensors.shape[0]
            # text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * currBatchSize).to(device)
            #Run OCR prediction
            if 'CTC' in opt.Prediction:
                preds = ocrModel(image_input_tensors, text_gt, is_train=False)
                preds_size = torch.IntTensor([preds.size(1)] * image_input_tensors.shape[0])
                _, preds_index = preds.max(2)
                preds_str_gt_1 = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = ocrModel(image_input_tensors, text_gt[:, :-1], is_train=False)  # align with Attention.forward
                _, preds_index = preds.max(2)
                preds_str_gt_1 = converter.decode(preds_index, length_for_pred)
                for idx, pred in enumerate(preds_str_gt_1):
                    pred_EOS = pred.find('[s]')
                    preds_str_gt_1[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

        
        
        for trImgCntr in range(image_input_tensors.shape[0]):
            #ocr accuracy
            # for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            c1_s1_input_gt = labels_gt[trImgCntr]
            c1_s1_input_ocr = preds_str_gt_1[trImgCntr]

            if c1_s1_input_gt == c1_s1_input_ocr:
                c1_s1_input_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(c1_s1_input_gt) == 0 or len(c1_s1_input_ocr) == 0:
                c1_s1_input_ed_correct += 0
            elif len(c1_s1_input_gt) > len(c1_s1_input_ocr):
                c1_s1_input_ed_correct += 1 - edit_distance(c1_s1_input_ocr, c1_s1_input_gt) / len(c1_s1_input_gt)
            else:
                c1_s1_input_ed_correct += 1 - edit_distance(c1_s1_input_ocr, c1_s1_input_gt) / len(c1_s1_input_ocr)
            
            evalCntr+=1
            
            
            fCntr += 1
    
    avg_c1_s1_input_wer = c1_s1_input_correct/float(evalCntr)
    avg_c1_s1_input_cer = c1_s1_input_ed_correct/float(evalCntr)

    # if not(opt.realVaData):
    with open(os.path.join(opt.exp_dir,opt.exp_name,'log_test.txt'), 'a') as log:
        # training loss and validation loss
        
        loss_log = f'Word Acc: {avg_c1_s1_input_wer:0.5f}, Test Input Char Acc: {avg_c1_s1_input_cer:0.5f}'
        
        print(loss_log)
        log.write(loss_log+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp18/', help='Where to store logs and models')
    parser.add_argument('--exp_name', default='debug', help='Where to store logs and models')
    parser.add_argument('--exp_iter', default='0', help='Where to store logs and models')
    parser.add_argument('--test_data', required=True, help='path to validation dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--saved_ocr_model', default='', help="path to model to continue training")
    
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
    parser.add_argument('--ocr_imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--ocr_imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--imgH_filt', type=int, default=48, help='the height of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--pairText', action='store_true', help='use additional text for generation')
    parser.add_argument('--lexFile', default='/checkpoint/pkrishnan/datasets/vocab/english-words.txt', help='unqiue words in language')
    
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,  help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, 
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--ocr_input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--char_embed_size', type=int, default=60, help='character embedding for content encoder')
    parser.add_argument('--vggNoMean', action='store_true', help='if yes; No mean normalization is applied to vgg input')
    parser.add_argument('--styleDetach', action='store_true', help='whether to detach style')
    parser.add_argument('--gan_type', default='lsgan', help='lsgan/nsgan/wgan')
    parser.add_argument('--style_input', action='store_true', help='whether target style input is given for training/validation')
    parser.add_argument('--style_content_input', action='store_true', help='whether target  input content image is given for training/validation')
    parser.add_argument('--styleNorm', default='bn', help='bn/in')
    parser.add_argument('--contentNorm', default='bn', help='bn/in')
    parser.add_argument('--sizeFilt', action='store_true', help='for applying size based image filtering')

    parser.add_argument('--debugFlag', action='store_true', help='for debugging')
    parser.add_argument('--visFlag', action='store_true', help='for visualization')

    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--input_latent", action="store_true")
    parser.add_argument("--content_inject_index", type=int, default=1)
    parser.add_argument("--zAlone", action="store_true", help="test original style GAN")
    parser.add_argument("--noiseConcat", action="store_true", help="noise concat with style")

    parser.add_argument('--cEncode', default='mlp', help='mlp/cnn')
    parser.add_argument('--taskActivation', default=None, help='None/tanh')
    parser.add_argument("--numLatents", type=int, default=1)
    parser.add_argument("--realTrData", action="store_true", help="flag for training real data where we don't have target style GT")
    parser.add_argument("--realVaData", action="store_true", help="flag for validation/testing real data where we don't have target style GT")

    
    parser.add_argument('--outPairFile', default="", help='unqiue words in language')

    opt = parser.parse_args()

    if opt.rgb:
        opt.input_channel = 3
    
    if opt.taskActivation == 'None':
        opt.taskActivation = None

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name), exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True

    test(opt)

