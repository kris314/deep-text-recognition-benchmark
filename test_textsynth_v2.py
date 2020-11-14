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

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignPairCollate, AlignPairImgCollate, AlignPairImgCollate_Test, AlignSynthTextCollate, Batch_Balanced_Dataset, tensor2im, save_image, phoc_gen, text_gen, text_gen_synth, LmdbStyleDataset, LmdbTestStyleContentDataset
from model import ModelV1, GlobalContentEncoder, VGGPerceptualEmbedLossModel, VGGFontModel
from modules.feature_extraction import ResNet_StyleExtractor, VGG_ContentExtractor, ResNet_StyleExtractor_WIN

import tflib as lib
import tflib.plot

sys.path.append('/private/home/pkrishnan/codes/st-scribe/stylegan2-pytorch/')

from model_word import GeneratorM2V4_2 as styleGANGen 
from model_word import EncDiscriminator as styleGANDis  
from non_leaking import augment
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader, data_sampler, is_distributed):
    epochCntr=0
    while True:
        if is_distributed:
            data_sampler.set_epoch(epochCntr)

        for batch in loader:
            yield batch
        epochCntr+=1



def make_noise(z_code, batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.cat((z_code, torch.randn(batch, latent_dim, device=device)),1)

    noises = torch.cat((z_code.repeat(2,1,1), torch.randn(n_noise, batch, latent_dim, device=device)),2).unbind(0)

    return noises

def make_noise_style(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises

def mixing_noise(batch, latent_dim, prob, device, z_code=None):
    if prob > 0 and random.random() < prob:
        if z_code is None:
            return make_noise_style(batch, latent_dim, 2, device)
        else:
            return make_noise(z_code, batch, latent_dim, 2, device)
    else:
        if z_code is None:
            return [make_noise_style(batch, latent_dim, 1, device)]
        else:
            return [make_noise(z_code, batch, latent_dim, 1, device)]

def mixing_noise_style(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise_style(batch, latent_dim, 2, device)

    else:
        return [make_noise_style(batch, latent_dim, 1, device)]

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

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

    # AlignCollate_valid = AlignPairImgCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    AlignCollate_valid = AlignPairImgCollate_Test(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    
    valid_dataset = LmdbTestStyleContentDataset(root=opt.test_data, opt=opt, dataMode=opt.realVaData)
    test_data_sampler = data_sampler(valid_dataset, shuffle=False, distributed=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, 
        shuffle=False,  # 'True' to check training progress with validation function.
        sampler=test_data_sampler,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True, drop_last=False)
    
    print('-' * 80)

    AlignCollate_text = AlignSynthTextCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    text_dataset = text_gen_synth(opt)
    text_data_sampler = data_sampler(text_dataset, shuffle=True, distributed=False)
    text_loader = torch.utils.data.DataLoader(
        text_dataset, batch_size=opt.batch_size,
        shuffle=False,
        sampler=text_data_sampler,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_text,
        drop_last=True)
    opt.num_class = len(converter.character)

 
    text_loader = sample_data(text_loader, text_data_sampler, False)

    c_code_size = opt.latent
    if opt.cEncode == 'mlp':
        cEncoder = GlobalContentEncoder(opt.num_class, text_len, opt.char_embed_size, c_code_size).to(device)
    elif opt.cEncode == 'cnn':
        if opt.contentNorm == 'in':
            cEncoder = ResNet_StyleExtractor_WIN(1, opt.latent).to(device)
        else:
            cEncoder = ResNet_StyleExtractor(1, opt.latent).to(device)
    if opt.styleNorm == 'in':
        styleModel = ResNet_StyleExtractor_WIN(opt.input_channel, opt.latent).to(device)
    else:
        styleModel = ResNet_StyleExtractor(opt.input_channel, opt.latent).to(device)
    
    ocrModel = ModelV1(opt).to(device)

    g_ema = styleGANGen(opt.size, opt.latent, opt.latent, opt.n_mlp, content_dim=c_code_size, channel_multiplier=opt.channel_multiplier).to(device)
    g_ema.eval()
    
    bestModelError=1e5

    ## Loading pre-trained files
    print(f'loading pretrained ocr model from {opt.saved_ocr_model}')
    checkpoint = torch.load(opt.saved_ocr_model, map_location=lambda storage, loc: storage)
    ocrModel.load_state_dict(checkpoint)

    print(f'loading pretrained synth model from {opt.saved_synth_model}')
    checkpoint = torch.load(opt.saved_synth_model, map_location=lambda storage, loc: storage)
    
    cEncoder.load_state_dict(checkpoint['cEncoder'])
    styleModel.load_state_dict(checkpoint['styleModel'])
    g_ema.load_state_dict(checkpoint['g_ema'])
     
    
    iCntr=0
    evalCntr=0
    fCntr=0
    
    valMSE=0.0
    valSSIM=0.0
    valPSNR=0.0
    c1_s1_input_correct=0.0
    c2_s1_gen_correct=0.0
    c1_s1_input_ed_correct=0.0
    c2_s1_gen_ed_correct=0.0
    
    
    ims, txts = [], []
    
    for vCntr, (image_input_tensors, image_output_tensors, labels_gt, labels_z_c, labelSynthImg, synth_z_c, input_1_shape, input_2_shape) in enumerate(valid_loader):
        print(vCntr)

        if opt.debugFlag and vCntr >10:
            break  
        
        image_input_tensors = image_input_tensors.to(device)
        image_output_tensors = image_output_tensors.to(device)

        if opt.realVaData and opt.outPairFile=="":
            # pdb.set_trace()
            labels_z_c, synth_z_c = next(text_loader)
        
        labelSynthImg = labelSynthImg.to(device)
        synth_z_c = synth_z_c.to(device)
        synth_z_c = synth_z_c[:labelSynthImg.shape[0]]
        
        text_z_c, length_z_c = converter.encode(labels_z_c, batch_max_length=opt.batch_max_length)
        text_gt, length_gt = converter.encode(labels_gt, batch_max_length=opt.batch_max_length)
        
        # print(labels_z_c)

        cEncoder.eval()
        styleModel.eval()
        g_ema.eval()

        with torch.no_grad():
            if opt.cEncode == 'mlp':    
                z_c_code = cEncoder(text_z_c)
                z_gt_code = cEncoder(text_gt)
            elif opt.cEncode == 'cnn':    
                z_c_code = cEncoder(synth_z_c)
                z_gt_code = cEncoder(labelSynthImg)
                
            style = styleModel(image_input_tensors)
            
            if opt.noiseConcat or opt.zAlone:
                style = mixing_noise(opt.batch_size, opt.latent, opt.mixing, device, style)
            else:
                style = [style]
            
            fake_img_c1_s1, _ = g_ema(style, z_gt_code, input_is_latent=opt.input_latent)
            fake_img_c2_s1, _ = g_ema(style, z_c_code, input_is_latent=opt.input_latent)

            currBatchSize = fake_img_c1_s1.shape[0]
            # text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * currBatchSize).to(device)
            #Run OCR prediction
            if 'CTC' in opt.Prediction:
                preds = ocrModel(fake_img_c1_s1, text_gt, is_train=False, inAct = opt.taskActivation)
                preds_size = torch.IntTensor([preds.size(1)] * currBatchSize)
                _, preds_index = preds.max(2)
                preds_str_fake_img_c1_s1 = converter.decode(preds_index.data, preds_size.data)

                preds = ocrModel(fake_img_c2_s1, text_z_c, is_train=False, inAct = opt.taskActivation)
                preds_size = torch.IntTensor([preds.size(1)] * currBatchSize)
                _, preds_index = preds.max(2)
                preds_str_fake_img_c2_s1 = converter.decode(preds_index.data, preds_size.data)

                preds = ocrModel(image_input_tensors, text_gt, is_train=False)
                preds_size = torch.IntTensor([preds.size(1)] * image_input_tensors.shape[0])
                _, preds_index = preds.max(2)
                preds_str_gt_1 = converter.decode(preds_index.data, preds_size.data)

                preds = ocrModel(image_output_tensors, text_z_c, is_train=False)
                preds_size = torch.IntTensor([preds.size(1)] * image_output_tensors.shape[0])
                _, preds_index = preds.max(2)
                preds_str_gt_2 = converter.decode(preds_index.data, preds_size.data)

            else:
                
                preds = ocrModel(fake_img_c1_s1, text_gt[:, :-1], is_train=False, inAct = opt.taskActivation)  # align with Attention.forward
                _, preds_index = preds.max(2)
                preds_str_fake_img_c1_s1 = converter.decode(preds_index, length_for_pred)
                for idx, pred in enumerate(preds_str_fake_img_c1_s1):
                    pred_EOS = pred.find('[s]')
                    preds_str_fake_img_c1_s1[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                
                preds = ocrModel(fake_img_c2_s1, text_z_c[:, :-1], is_train=False, inAct = opt.taskActivation)  # align with Attention.forward
                _, preds_index = preds.max(2)
                preds_str_fake_img_c2_s1 = converter.decode(preds_index, length_for_pred)
                for idx, pred in enumerate(preds_str_fake_img_c2_s1):
                    pred_EOS = pred.find('[s]')
                    preds_str_fake_img_c2_s1[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                

                preds = ocrModel(image_input_tensors, text_gt[:, :-1], is_train=False)  # align with Attention.forward
                _, preds_index = preds.max(2)
                preds_str_gt_1 = converter.decode(preds_index, length_for_pred)
                for idx, pred in enumerate(preds_str_gt_1):
                    pred_EOS = pred.find('[s]')
                    preds_str_gt_1[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

                preds = ocrModel(image_output_tensors, text_z_c[:, :-1], is_train=False)  # align with Attention.forward
                _, preds_index = preds.max(2)
                preds_str_gt_2 = converter.decode(preds_index, length_for_pred)
                for idx, pred in enumerate(preds_str_gt_2):
                    pred_EOS = pred.find('[s]')
                    preds_str_gt_2[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

        pathPrefix = os.path.join(opt.valDir, opt.exp_iter)
        os.makedirs(os.path.join(pathPrefix), exist_ok=True)
        
        for trImgCntr in range(image_output_tensors.shape[0]):
            
            if opt.outPairFile!="":
                labelId = 'label-' + valid_loader.dataset.pairId[fCntr] + '-' + str(fCntr)
            else:
                labelId = 'label-%09d' % valid_loader.dataset.filtered_index_list[fCntr]
            #evaluations
            valRange = (-1,+1)
            # pdb.set_trace()
            # inpTensor = skimage.img_as_ubyte(resize(tensor2im(image_input_tensors[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0])))
            # gtTensor = skimage.img_as_ubyte(resize(tensor2im(image_output_tensors[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0])))
            # predTensor = skimage.img_as_ubyte(resize(tensor2im(fake_img_c2_s1[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0])))
            
            # inpTensor = resize(tensor2im(image_input_tensors[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0]), anti_aliasing=True)
            # gtTensor = resize(tensor2im(image_output_tensors[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0]), anti_aliasing=True)
            # predTensor = resize(tensor2im(fake_img_c2_s1[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0]), anti_aliasing=True)

            inpTensor = F.interpolate(image_input_tensors[trImgCntr].unsqueeze(0),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0]))
            gtTensor = F.interpolate(image_output_tensors[trImgCntr].unsqueeze(0),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0]))
            predTensor = F.interpolate(fake_img_c2_s1[trImgCntr].unsqueeze(0),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0]))
            predGTTensor = F.interpolate(fake_img_c1_s1[trImgCntr].unsqueeze(0),(input_1_shape[trImgCntr][1], input_1_shape[trImgCntr][0]))

            # inpTensor = cv2.resize(tensor2im(image_input_tensors[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),tuple(input_1_shape[trImgCntr]))
            # gtTensor = cv2.resize(tensor2im(image_output_tensors[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),tuple(input_1_shape[trImgCntr]))
            # predTensor = cv2.resize(tensor2im(fake_img_c2_s1[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),tuple(input_1_shape[trImgCntr]))
            # predGTTensor = cv2.resize(tensor2im(fake_img_c1_s1[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),tuple(input_1_shape[trImgCntr]))

            # inpTensor = cv2.medianBlur(cv2.resize(tensor2im(image_input_tensors[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),tuple(input_1_shape[trImgCntr])),5)
            # gtTensor = cv2.medianBlur(cv2.resize(tensor2im(image_output_tensors[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),tuple(input_1_shape[trImgCntr])),5)
            # predTensor = cv2.medianBlur(cv2.resize(tensor2im(fake_img_c2_s1[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1])),tuple(input_1_shape[trImgCntr])),5)
            # pdb.set_trace()

            if not(opt.realVaData):
                evalMSE = mean_squared_error(tensor2im(gtTensor.squeeze())/255, tensor2im(predTensor.squeeze())/255)
                # evalMSE = mean_squared_error(gtTensor/255, predTensor/255)
                evalSSIM = structural_similarity(tensor2im(gtTensor.squeeze())/255, tensor2im(predTensor.squeeze())/255, multichannel=True)
                # evalSSIM = structural_similarity(gtTensor/255, predTensor/255, multichannel=True)
                evalPSNR = peak_signal_noise_ratio(tensor2im(gtTensor.squeeze())/255, tensor2im(predTensor.squeeze())/255)
                # evalPSNR = peak_signal_noise_ratio(gtTensor/255, predTensor/255)
                # print(evalMSE,evalSSIM,evalPSNR)

                valMSE+=evalMSE
                valSSIM+=evalSSIM
                valPSNR+=evalPSNR

            #ocr accuracy
            # for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            c1_s1_input_gt = labels_gt[trImgCntr]
            c1_s1_input_ocr = preds_str_gt_1[trImgCntr]
            c2_s1_gen_gt = labels_z_c[trImgCntr]
            c2_s1_gen_ocr = preds_str_fake_img_c2_s1[trImgCntr]

            if c1_s1_input_gt == c1_s1_input_ocr:
                c1_s1_input_correct += 1
            if c2_s1_gen_gt == c2_s1_gen_ocr:
                c2_s1_gen_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(c1_s1_input_gt) == 0 or len(c1_s1_input_ocr) == 0:
                c1_s1_input_ed_correct += 0
            elif len(c1_s1_input_gt) > len(c1_s1_input_ocr):
                c1_s1_input_ed_correct += 1 - edit_distance(c1_s1_input_ocr, c1_s1_input_gt) / len(c1_s1_input_gt)
            else:
                c1_s1_input_ed_correct += 1 - edit_distance(c1_s1_input_ocr, c1_s1_input_gt) / len(c1_s1_input_ocr)
            
            if len(c2_s1_gen_gt) == 0 or len(c2_s1_gen_ocr) == 0:
                c2_s1_gen_ed_correct += 0
            elif len(c2_s1_gen_gt) > len(c2_s1_gen_ocr):
                c2_s1_gen_ed_correct += 1 - edit_distance(c2_s1_gen_ocr, c2_s1_gen_gt) / len(c2_s1_gen_gt)
            else:
                c2_s1_gen_ed_correct += 1 - edit_distance(c2_s1_gen_ocr, c2_s1_gen_gt) / len(c2_s1_gen_ocr)
            
            evalCntr+=1
            
            #save generated images
            if opt.visFlag and iCntr>500:
                pass
            else:
                try:
                    if iCntr == 0:
                        # update website
                        webpage = html.HTML(pathPrefix, 'Experiment name = %s' % 'Test')
                        webpage.add_header('Testing iteration')
                        
                    iCntr += 1
                    img_path_c1_s1 = os.path.join(labelId+'_pred_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr_'+preds_str_fake_img_c1_s1[trImgCntr]+'.png')
                    img_path_gt_1 = os.path.join(labelId+'_gt_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr_'+preds_str_gt_1[trImgCntr]+'.png')
                    img_path_gt_2 = os.path.join(labelId+'_gt_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr_'+preds_str_gt_2[trImgCntr]+'.png')
                    img_path_c2_s1 = os.path.join(labelId+'_pred_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr_'+preds_str_fake_img_c2_s1[trImgCntr]+'.png')
                    
                    ims.append([img_path_gt_1, img_path_c1_s1, img_path_gt_2, img_path_c2_s1])

                    content_c1_s1 = 'PSTYLE-1 '+'Text-1:' + labels_gt[trImgCntr]+' OCR:' + preds_str_fake_img_c1_s1[trImgCntr]
                    content_gt_1 = 'OSTYLE-1 '+'GT:' + labels_gt[trImgCntr]+' OCR:' + preds_str_gt_1[trImgCntr]
                    content_gt_2 = 'OSTYLE-1 '+'GT:' + labels_z_c[trImgCntr]+' OCR:'+preds_str_gt_2[trImgCntr]
                    content_c2_s1 = 'PSTYLE-1 '+'Text-2:' + labels_z_c[trImgCntr]+' OCR:'+preds_str_fake_img_c2_s1[trImgCntr]
                    
                    txts.append([content_gt_1, content_c1_s1, content_gt_2, content_c2_s1])
                    
                    utils.save_image(predGTTensor,os.path.join(pathPrefix,labelId+'_pred_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr_'+preds_str_fake_img_c1_s1[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                    # cv2.imwrite(os.path.join(pathPrefix,labelId+'_pred_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr_'+preds_str_fake_img_c1_s1[trImgCntr]+'.png'), predGTTensor)
                    
                    # pdb.set_trace()
                    if not opt.zAlone:
                        utils.save_image(inpTensor,os.path.join(pathPrefix,labelId+'_gt_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr_'+preds_str_gt_1[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                        utils.save_image(gtTensor,os.path.join(pathPrefix,labelId+'_gt_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr_'+preds_str_gt_2[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                        utils.save_image(predTensor,os.path.join(pathPrefix,labelId+'_pred_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr_'+preds_str_fake_img_c2_s1[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1)) 
                        
                        # imsave(os.path.join(pathPrefix,labelId+'_gt_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr_'+preds_str_gt_1[trImgCntr]+'.png'), inpTensor)
                        # imsave(os.path.join(pathPrefix,labelId+'_gt_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr_'+preds_str_gt_2[trImgCntr]+'.png'), gtTensor)
                        # imsave(os.path.join(pathPrefix,labelId+'_pred_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr_'+preds_str_fake_img_c2_s1[trImgCntr]+'.png'), predTensor) 

                        # cv2.imwrite(os.path.join(pathPrefix,labelId+'_gt_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr_'+preds_str_gt_1[trImgCntr]+'.png'), inpTensor)
                        # cv2.imwrite(os.path.join(pathPrefix,labelId+'_gt_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr_'+preds_str_gt_2[trImgCntr]+'.png'), gtTensor)
                        # cv2.imwrite(os.path.join(pathPrefix,labelId+'_pred_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr_'+preds_str_fake_img_c2_s1[trImgCntr]+'.png'), predTensor) 
                except:
                    print('Warning while saving validation image')
            
            fCntr += 1
    
    webpage.add_images(ims, txts, width=256, realFlag=opt.realVaData)    
    webpage.save()
    
    avg_valMSE = valMSE/float(evalCntr)
    avg_valSSIM = valSSIM/float(evalCntr)
    avg_valPSNR = valPSNR/float(evalCntr)
    avg_c1_s1_input_wer = c1_s1_input_correct/float(evalCntr)
    avg_c2_s1_gen_wer = c2_s1_gen_correct/float(evalCntr)
    avg_c1_s1_input_cer = c1_s1_input_ed_correct/float(evalCntr)
    avg_c2_s1_gen_cer = c2_s1_gen_ed_correct/float(evalCntr)

    # if not(opt.realVaData):
    with open(os.path.join(opt.exp_dir,opt.exp_name,'log_test.txt'), 'a') as log:
        # training loss and validation loss
        if opt.realVaData:
            loss_log = f'Test Input Word Acc: {avg_c1_s1_input_wer:0.5f}, Test Gen Word Acc: {avg_c2_s1_gen_wer:0.5f}, Test Input Char Acc: {avg_c1_s1_input_cer:0.5f}, Test Gen Char Acc: {avg_c2_s1_gen_cer:0.5f}'
        else:
            loss_log = f'Test MSE: {avg_valMSE:0.5f}, Test SSIM: {avg_valSSIM:0.5f}, Test PSNR: {avg_valPSNR:0.5f}, Test Input Word Acc: {avg_c1_s1_input_wer:0.5f}, Test Gen Word Acc: {avg_c2_s1_gen_wer:0.5f}, Test Input Char Acc: {avg_c1_s1_input_cer:0.5f}, Test Gen Char Acc: {avg_c2_s1_gen_cer:0.5f}'
        
        print(loss_log)
        log.write(loss_log+"\n")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp18/', help='Where to store logs and models')
    parser.add_argument('--exp_name', default='debug', help='Where to store logs and models')
    parser.add_argument('--exp_iter', default='0', help='Where to store logs and models')
    parser.add_argument('--test_data', required=True, help='path to validation dataset')
    parser.add_argument('--words_file', required=True, default='', help="path to words file. phoc will be sampled from this file")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--saved_ocr_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_synth_model', default='', help="path to model to continue training")
    
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
    opt.valDir=os.path.join(opt.exp_dir,opt.exp_name,'testImages')    

    os.makedirs(opt.valDir, exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True

    test(opt)

