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

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignPairCollate, AlignPairImgCollate, AlignSynthTextCollate, Batch_Balanced_Dataset, tensor2im, save_image, phoc_gen, text_gen, text_gen_synth, LmdbStyleDataset, LmdbStyleContentDataset
# from model import ModelV1, StyleTensorEncoder, StyleLatentEncoder, MsImageDisV2, AdaIN_Tensor_WordGenerator, VGGPerceptualLossModel, Mixer
from model import ModelV1, GlobalContentEncoder, VGGPerceptualEmbedLossModel, VGGFontModel
# from test_synth import validation_synth_v7
from modules.feature_extraction import ResNet_StyleExtractor, VGG_ContentExtractor, ResNet_StyleExtractor_WIN, ResNet_StyleExtractor_FPN

import tflib as lib
import tflib.plot

sys.path.append('/private/home/pkrishnan/codes/st-scribe/stylegan2-pytorch/')

try:
    import wandb

except ImportError:
    wandb = None

from model_word import GeneratorM2V4_7 as styleGANGen 
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


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


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

def train(opt):
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

    log = open(os.path.join(opt.exp_dir,opt.exp_name,'log_dataset.txt'), 'a')
    AlignCollate_valid = AlignPairImgCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    # train_dataset = LmdbStyleDataset(root=opt.train_data, opt=opt)
    
    train_dataset = LmdbStyleContentDataset(root=opt.train_data, opt=opt, dataMode=opt.realTrData)
    train_data_sampler = data_sampler(train_dataset, shuffle=True, distributed=opt.distributed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, 
        shuffle=False,  # 'True' to check training progress with validation function.
        sampler=train_data_sampler,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True, drop_last=True)
    train_loader = sample_data(train_loader, train_data_sampler, opt.distributed)
    print('-' * 80)
    
    # valid_dataset = LmdbStyleDataset(root=opt.valid_data, opt=opt)
    valid_dataset = LmdbStyleContentDataset(root=opt.valid_data, opt=opt, dataMode=opt.realVaData)
    test_data_sampler = data_sampler(valid_dataset, shuffle=False, distributed=opt.distributed)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, 
        shuffle=False,  # 'True' to check training progress with validation function.
        sampler=test_data_sampler,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True, drop_last=True)
    numTestSamples = len(valid_loader)
    # valid_loader = sample_data(valid_loader)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    AlignCollate_text = AlignSynthTextCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    # text_dataset = text_gen(opt)
    text_dataset = text_gen_synth(opt)
    text_data_sampler = data_sampler(text_dataset, shuffle=True, distributed=opt.distributed)
    text_loader = torch.utils.data.DataLoader(
        text_dataset, batch_size=opt.batch_size,
        shuffle=False,
        sampler=text_data_sampler,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_text,
        drop_last=True)
    opt.num_class = len(converter.character)

    text_loader = sample_data(text_loader, text_data_sampler, opt.distributed)

    if not opt.zAlone:
        c_code_size = opt.latent
        if opt.cEncode == 'mlp':
            cEncoder = GlobalContentEncoder(opt.num_class, text_len, opt.char_embed_size, c_code_size).to(device)
        elif opt.cEncode == 'cnn':
            # for synthetic image
            # cEncoder = VGG_ContentExtractor(1, opt.latent).to(device)
            if opt.contentNorm == 'in':
                cEncoder = ResNet_StyleExtractor_WIN(1, opt.latent).to(device)
            else:
                cEncoder = ResNet_StyleExtractor(1, opt.latent).to(device)
        if opt.styleNorm == 'in':
            styleModel = ResNet_StyleExtractor_WIN(opt.input_channel, opt.style_latent).to(device)
        else:
            # styleModel = ResNet_StyleExtractor(opt.input_channel, opt.style_latent).to(device)
            styleModel = ResNet_StyleExtractor_FPN(opt.input_channel, opt.style_latent).to(device)
        ocrModel = ModelV1(opt).to(device)

        # #temp
        # pdb.set_trace()
        # ocrModel = torch.nn.DataParallel(ocrModel).to(device)
        # checkpoint = torch.load(opt.saved_ocr_model)
        # ocrModel.load_state_dict(checkpoint)
        # torch.save(ocrModel.module.state_dict(),'/checkpoint/pkrishnan/experiments/scribe/pretrained/TPS-ResNet-BiLSTM-Attn-case-sensitive_actual_nonparallel.pth')
        
        if 'CTC' in opt.Prediction:
            ocrCriterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        else:
            ocrCriterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
        
        

        if opt.imgReconLoss == 'l1':
            reconCriterion = torch.nn.L1Loss()
        elif opt.imgReconLoss == 'l2':
            reconCriterion = torch.nn.MSELoss()

        if opt.saved_font_model !='' and opt.saved_font_model !='None':
            checkpoint = torch.load(opt.saved_font_model, map_location=lambda storage, loc: storage)
            preTrainedVGGModel = VGGFontModel(models.vgg19(pretrained=False), numClasses=checkpoint['vggFontModel']['classifier.6.weight'].shape[0])
            preTrainedVGGModel.load_state_dict(checkpoint['vggFontModel'])
            resize = False
        else:
            preTrainedVGGModel = models.vgg19(pretrained=True)
            resize = True

        vggModel = VGGPerceptualEmbedLossModel(preTrainedVGGModel, reconCriterion, resize).to(device)
        vggModel.eval()
    else:
        c_code_size = 0

    
    # weight initialization for style and content models
    if not opt.zAlone:
        for model in [styleModel, cEncoder]:
            for name, param in model.named_parameters():
                if 'localization_fc2' in name:
                    print(f'Skip {name} as it is already initialized')
                    continue
                try:
                    if 'bias' in name:
                        init.constant_(param, 0.0)
                    elif 'weight' in name:
                        init.kaiming_normal_(param)
                except Exception as e:  # for batchnorm.
                    print('Exception in weight init'+ name)
                    if 'weight' in name:
                        param.data.fill_(1)
                    continue
    if opt.noiseConcat:
        genModel = styleGANGen(opt.size, opt.style_latent*2, opt.latent, opt.n_mlp, content_dim=c_code_size, channel_multiplier=opt.channel_multiplier).to(device)
        g_ema = styleGANGen(opt.size, opt.style_latent*2, opt.latent, opt.n_mlp, content_dim=c_code_size, channel_multiplier=opt.channel_multiplier).to(device)
    else:
        genModel = styleGANGen(opt.size, opt.style_latent, opt.latent, opt.n_mlp, content_dim=c_code_size, channel_multiplier=opt.channel_multiplier).to(device)
        g_ema = styleGANGen(opt.size, opt.style_latent, opt.latent, opt.n_mlp, content_dim=c_code_size, channel_multiplier=opt.channel_multiplier).to(device)
    g_ema.eval()
    disEncModel = styleGANDis(opt.size, channel_multiplier=opt.channel_multiplier, input_dim=opt.input_channel, code_s_dim=opt.latent).to(device)
    
    accumulate(g_ema, genModel, 0)
    
    g_reg_ratio = opt.g_reg_every / (opt.g_reg_every + 1)
    d_reg_ratio = opt.d_reg_every / (opt.d_reg_every + 1)

    if opt.zAlone:
        optimizer = optim.Adam(
            genModel.parameters(),
            lr=opt.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
    else:
        optimizer = optim.Adam(
            list(genModel.parameters())+list(cEncoder.parameters())+list(styleModel.parameters()),
            lr=opt.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        if opt.optim == "adam":
            ocr_optimizer = optim.Adam(
                ocrModel.parameters(),
                lr=opt.lr,
                betas=(0.9, 0.99),
            )
        else:
            ocr_optimizer = optim.Adadelta(
                ocrModel.parameters(),
                lr=1.0,
                rho=opt.rho, eps=opt.eps,
            )
    dis_optimizer = optim.Adam(
        disEncModel.parameters(),
        lr=opt.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    # print('Model Initialization')
    
    bestModelError=1e5

    ## Loading pre-trained files
    if opt.modelFolderFlag:
        if len(glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth")))>0:
            opt.saved_synth_model = glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth"))[-1]

    if not opt.zAlone:
        if opt.saved_ocr_model !='' and opt.saved_ocr_model !='None':
            print(f'loading pretrained ocr model from {opt.saved_ocr_model}')
            checkpoint = torch.load(opt.saved_ocr_model, map_location=lambda storage, loc: storage)
            ocrModel.load_state_dict(checkpoint)
    
    if opt.saved_gen_model !='' and opt.saved_gen_model !='None':
        print(f'loading pretrained gen model from {opt.saved_gen_model}')
        checkpoint = torch.load(opt.saved_gen_model, map_location=lambda storage, loc: storage)
        genModel.load_state_dict(checkpoint['genModel'])
        g_ema.load_state_dict(checkpoint['g_ema'])

    if opt.saved_synth_model != '' and opt.saved_synth_model != 'None':
        print(f'loading pretrained synth model from {opt.saved_synth_model}')
        checkpoint = torch.load(opt.saved_synth_model, map_location=lambda storage, loc: storage)
        
        if not opt.zAlone:
            cEncoder.load_state_dict(checkpoint['cEncoder'])
            styleModel.load_state_dict(checkpoint['styleModel'])
            ocrModel.load_state_dict(checkpoint['ocrModel'])
        genModel.load_state_dict(checkpoint['genModel'])
        g_ema.load_state_dict(checkpoint['g_ema'])
        disEncModel.load_state_dict(checkpoint['disEncModel'])
        
        optimizer.load_state_dict(checkpoint["optimizer"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        if not opt.zAlone:
            ocr_optimizer.load_state_dict(checkpoint["ocr_optimizer"])
        if 'bestModelError' in checkpoint:
            bestModelError = checkpoint['bestModelError']
    # print('Loaded checkpoint')
    
    if not opt.zAlone and opt.distributed:

        cEncoder = torch.nn.parallel.DistributedDataParallel(
            cEncoder,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        # cEncoder= torch.nn.DataParallel(cEncoder).to(device)
        cEncoder.train()

        styleModel = torch.nn.parallel.DistributedDataParallel(
            styleModel,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        # styleModel= torch.nn.DataParallel(styleModel).to(device)
        styleModel.train()

        vggModel = torch.nn.parallel.DistributedDataParallel(
            vggModel,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        # vggModel= torch.nn.DataParallel(vggModel).to(device)
        vggModel.eval()

        ocrModel = torch.nn.parallel.DistributedDataParallel(
            ocrModel,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        # ocrModel = torch.nn.DataParallel(ocrModel).to(device)
        if opt.ocrFixed:
            if opt.Transformation == 'TPS':
                ocrModel.module.Transformation.eval()
            ocrModel.module.FeatureExtraction.eval()
            ocrModel.module.AdaptiveAvgPool.eval()
            # ocrModel.module.SequenceModeling.eval()
            ocrModel.module.Prediction.eval()
        else:
            ocrModel.train()

    if opt.distributed:
        genModel = torch.nn.parallel.DistributedDataParallel(
            genModel,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        disEncModel = torch.nn.parallel.DistributedDataParallel(
            disEncModel,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
    # genModel = torch.nn.DataParallel(genModel).to(device)
    # g_ema = torch.nn.DataParallel(g_ema).to(device)
    # genModel.train()
    # g_ema.eval()
    
    # disEncModel = torch.nn.DataParallel(disEncModel).to(device)
    # disEncModel.train()
    # print('Loaded distributed')
    
    if opt.distributed:
        if not opt.zAlone:
            cEncoder_module = cEncoder.module
            styleModel_module = styleModel.module
            ocrModel_module = ocrModel.module
        genModel_module = genModel.module
        disEncModel_module = disEncModel.module

    else:
        if not opt.zAlone:
            cEncoder_module = cEncoder
            styleModel_module = styleModel
            ocrModel_module = ocrModel
        genModel_module = genModel
        disEncModel_module = disEncModel

    # print('Loading module')
    
    # loss averager
    loss_recon_train = Averager()
    loss_recon_val = Averager()
    loss_avg_dis = Averager()
    loss_avg_gen = Averager()
    log_r1_val = Averager()
    log_avg_path_loss_val = Averager()
    log_avg_mean_path_length_avg = Averager()
    log_ada_aug_p = Averager()
    loss_avg_ocr_sup = Averager()
    loss_avg_ocr_unsup = Averager()
    loss_avg_style_ucode = Averager()
    loss_avg_style_scode = Averager()
    loss_avg_img_recon = Averager()
    loss_avg_cycle_recon = Averager()
    loss_avg_vgg_per = Averager()
    loss_avg_vgg_sty = Averager()
    loss_avg_vgg_emb = Averager()

    """ final options """
    with open(os.path.join(opt.exp_dir,opt.exp_name,'opt.txt'), 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    
    if opt.saved_synth_model != '' and opt.saved_synth_model != 'None':
        try:
            start_iter = int(opt.saved_synth_model.split('_')[-2].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    
    #get schedulers
    scheduler = get_scheduler(optimizer,opt)
    dis_scheduler = get_scheduler(dis_optimizer,opt)
    if not opt.zAlone:
        ocr_scheduler = get_scheduler(ocr_optimizer,opt)

    iteration = start_iter
    cntr=0
    
    mean_path_length = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    # loss_dict = {}

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = opt.augment_p if opt.augment_p > 0 else 0.0
    ada_aug_step = opt.ada_target / opt.ada_length
    r_t_stat = 0
    epsilon = 10e-50
    sample_z = torch.randn(opt.n_sample, opt.latent, device=device)

    while(True):
        # print(cntr)
        # train part
        start_time = time.time()
        if not opt.testFlag:
            if opt.lr_policy !="None":
                scheduler.step()
                dis_scheduler.step()
                if not opt.zAlone:
                    ocr_scheduler.step()
            
            image_input_tensors, image_output_tensors, labels_gt, labels_z_c, labelSynthImg, synth_z_c  = next(train_loader)
            # labels_z_c, synth_z_c = next(text_loader)
            # labels_z_c, synth_z_c = labels_output, labelSynthImg_output
            
            # print(labels)
            # print(labels_z_c)
            
            image_input_tensors = image_input_tensors.to(device)
            image_output_tensors = image_output_tensors.to(device)
            # gt_image_tensors = image_input_tensors.detach()    #exemplar word style image; training OCR
            # real_image_tensors = image_input_tensors.detach()  #discriminator
            synth_z_c = synth_z_c.to(device)

            # labels_gt = labels[:opt.batch_size]
            
            if not opt.zAlone:
                requires_grad(cEncoder, False)
                requires_grad(styleModel, False)
                requires_grad(ocrModel, False)
                requires_grad(vggModel, False)
            requires_grad(genModel, False)
            requires_grad(disEncModel, True)
            

            text_z_c, length_z_c = converter.encode(labels_z_c, batch_max_length=opt.batch_max_length)
            text_gt, length_gt = converter.encode(labels_gt, batch_max_length=opt.batch_max_length)
            
            # print('Before cEncoder')
            
            if opt.zAlone:
                z_c_code = None
                style = None
            else:
                if opt.cEncode == 'mlp':    
                    z_c_code = cEncoder(text_z_c)
                elif opt.cEncode == 'cnn':
                    z_c_code = cEncoder(synth_z_c)
                # print('Before styleModel')
                
                style = styleModel(image_input_tensors)
            
            # if opt.noiseConcat or opt.zAlone:
            #     style = mixing_noise(opt.batch_size, opt.latent, opt.mixing, device, style)
            # else:
            #     style = [style]
            # print('Before genModel')
            
            fake_img,_ = genModel(style, z_c_code, input_is_latent=opt.input_latent)
            # print('After genModel')
              
            #unsupervised style code prediction on generated image using StyleEncoder/Discriminator
            if opt.gamma_e>0.0 and not opt.zAlone:
                uPred_style_code = disEncModel(fake_img, mode='enc')
                uCost = reconCriterion(uPred_style_code, style[0][:opt.latent])
            else:
                uCost = torch.tensor(0.0)


            #Domain discriminator
            # print('Before disModel')
               
            fake_pred = disEncModel(fake_img)
            real_pred = disEncModel(image_input_tensors)
            # print('After disModel')
            
            disCost = d_logistic_loss(real_pred, fake_pred)

            dis_t_cost = disCost + opt.gamma_e*uCost
            loss_avg_dis.add(disCost)
            loss_avg_style_ucode.add(uCost)

            disEncModel.zero_grad()
            dis_t_cost.backward()

            if opt.grad_clip !=0.0:
                torch.nn.utils.clip_grad_norm_(disEncModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            
            dis_optimizer.step()
            # print('After disOptim backward')
            

            d_regularize = cntr % opt.d_reg_every == 0


            if d_regularize:
                image_input_tensors.requires_grad = True
                # print('before d_regularize backward')
                
                real_pred = disEncModel(image_input_tensors)
                
                r1_loss = d_r1_loss(real_pred, image_input_tensors)

                disEncModel.zero_grad()
                (opt.r1 / 2 * r1_loss * opt.d_reg_every + 0 * real_pred[0]).backward()
                # print('after d_regularize backward')
                
                if opt.grad_clip !=0.0:
                    torch.nn.utils.clip_grad_norm_(disEncModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
                    

                dis_optimizer.step()
            log_r1_val.add(r1_loss)
            
            image_input_tensors.requires_grad = False
            # Recognizer update
            if not opt.ocrFixed and not opt.zAlone:
                
                requires_grad(disEncModel, False)
                if not opt.zAlone:
                    requires_grad(ocrModel, True)

                if 'CTC' in opt.Prediction:
                    preds_recon = ocrModel(image_input_tensors, text_gt, is_train=True, inAct = opt.taskActivation)
                    preds_size = torch.IntTensor([preds_recon.size(1)] * opt.batch_size)
                    preds_recon_softmax = preds_recon.log_softmax(2).permute(1, 0, 2)
                    ocrCost = ocrCriterion(preds_recon_softmax, text_gt, preds_size, length_gt)
                else:
                    preds_recon = ocrModel(image_input_tensors, text_gt[:, :-1], is_train=True, inAct = opt.taskActivation)  # align with Attention.forward
                    target = text_gt[:, 1:]  # without [GO] Symbol
                    ocrCost = ocrCriterion(preds_recon.view(-1, preds_recon.shape[-1]), target.contiguous().view(-1))
                    # print("Not implemented error")
                    # sys.exit()
                
                ocrModel.zero_grad()
                ocrCost.backward()
                if opt.grad_clip !=0.0:
                    torch.nn.utils.clip_grad_norm_(ocrModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
                ocr_optimizer.step()
            else:
                ocrCost = torch.tensor(0.0)

            loss_avg_ocr_sup.add(ocrCost)

            # [Word Generator] update
            image_input_tensors, image_output_tensors, labels_gt, labels_z_c, labelSynthImg, synth_z_c  = next(train_loader)
            # labels_z_c, synth_z_c = next(text_loader)
            # print(labels_z_c)

            image_input_tensors = image_input_tensors.to(device)
            image_output_tensors = image_output_tensors.to(device)
            # gt_image_tensors = image_input_tensors[:opt.batch_size].detach()    #exemplar word style image; training OCR
            # labels_gt = labels[:opt.batch_size]
            

            labelSynthImg = labelSynthImg.to(device)
            synth_z_c = synth_z_c.to(device)

            if not opt.zAlone:
                requires_grad(cEncoder, True)
                requires_grad(styleModel, True)
                requires_grad(ocrModel, False)
                requires_grad(vggModel, False)
            requires_grad(genModel, True)
            requires_grad(disEncModel, False)
            

            text_z_c, length_z_c = converter.encode(labels_z_c, batch_max_length=opt.batch_max_length)
            text_gt, length_gt = converter.encode(labels_gt, batch_max_length=opt.batch_max_length)

            # print('before generator cEncoder')
            
            if opt.zAlone:
                z_c_code = None
                style = None
            else: 
                if opt.cEncode == 'mlp':   
                    z_c_code = cEncoder(text_z_c)
                    z_gt_code = cEncoder(text_gt)
                elif opt.cEncode == 'cnn':
                    z_c_code = cEncoder(synth_z_c)
                    z_gt_code = cEncoder(labelSynthImg)
                # print('after generator cEncoder')
                
                style = styleModel(image_input_tensors)
                # print('after generator styleModel')
                

            # if opt.noiseConcat or opt.zAlone:
            #     style = mixing_noise(opt.batch_size, opt.latent, opt.mixing, device, style)
            # else:
            #     style = [style]
            
            # fake_img,_ = genModel(style, z_c_code, input_is_latent=opt.input_latent)
            fake_gt_img,_ = genModel(style, z_gt_code, input_is_latent=opt.input_latent)
            # print('after generator genModel')
            

            fake_pred = disEncModel(fake_gt_img)
            disGenCost = g_nonsaturating_loss(fake_pred)
            # print('after generator disModel')
            

            if opt.zAlone:
                ocrCost = torch.tensor(0.0)
                uCost = torch.tensor(0.0)
                imgReconCost = torch.tensor(0.0)
                vggPerCost = torch.tensor(0.0)
                vggStyCost = torch.tensor(0.0)
                vggEmbCost = torch.tensor(0.0)
            else:
                #Compute OCR prediction (Reconstruction of content)
                if 'CTC' in opt.Prediction:
                    preds_recon = ocrModel(fake_gt_img, text_gt, is_train=False, inAct = opt.taskActivation)
                    preds_size = torch.IntTensor([preds_recon.size(1)] * opt.batch_size)
                    preds_recon_softmax = preds_recon.log_softmax(2).permute(1, 0, 2)
                    ocrCost = ocrCriterion(preds_recon_softmax, text_gt, preds_size, length_gt)

                else:
                    preds_recon = ocrModel(fake_gt_img, text_gt[:, :-1], is_train=False, inAct = opt.taskActivation)  # align with Attention.forward
                    target = text_gt[:, 1:]  # without [GO] Symbol
                    ocrCost = ocrCriterion(preds_recon.view(-1, preds_recon.shape[-1]), target.contiguous().view(-1))
                    # print("Not implemented error")
                    # sys.exit()
                # print('after generator ocrModel')
                
                if opt.gamma_g>0.0:
                    uPred_style_code = disEncModel(fake_gt_img, mode='enc')
                    uCost = reconCriterion(uPred_style_code, style[0][:opt.latent])
                else:
                    uCost = torch.tensor(0.0)
                
                if opt.reconWeight>0.0:
                    imgReconCost = reconCriterion(fake_gt_img, image_input_tensors)
                else:
                    imgReconCost = torch.tensor(0.0)
                
                # print('after generator recon genModel')
                
                
                if opt.cycleReconWeight > 0.0:
                    
                    style_fake = styleModel(fake_gt_img)

                    # if opt.noiseConcat or opt.zAlone:
                    #     style_fake = mixing_noise(opt.batch_size, opt.latent, opt.mixing, device, style_fake)
                    # else:
                    #     style_fake = [style_fake]
                    fake_recon_img, _ = genModel(style_fake, z_gt_code, input_is_latent=opt.input_latent)
                    cycleReconCost = reconCriterion(fake_recon_img, image_input_tensors)
                else:
                    cycleReconCost = torch.tensor(0.0)
                
                # print('after generator cycle genModel')
                
                
                if opt.vggPerWeight>0.0 or opt.vggStyWeight>0.0 or opt.vggEmbWeight>0.0:
                    vggPerCost , vggStyCost, vggEmbCost  = vggModel(fake_gt_img, image_input_tensors, inAct=opt.taskActivation, normFlag=not(opt.vggNoMean))
                else:
                    vggPerCost = torch.tensor(0.0)
                    vggStyCost = torch.tensor(0.0)
                    vggEmbCost = torch.tensor(0.0)

                # print('after generator vggModel')
                

            genModel.zero_grad()
            if not opt.zAlone:
                styleModel.zero_grad()
                cEncoder.zero_grad()

            gen_enc_cost = disGenCost + opt.ocrWeight * ocrCost + opt.gamma_g * uCost \
                            + opt.reconWeight * imgReconCost +  opt.vggPerWeight * vggPerCost \
                            +  opt.vggStyWeight * vggStyCost +  opt.vggEmbWeight * vggEmbCost \
                            + opt.cycleReconWeight * cycleReconCost

            gen_enc_cost.backward()
            # print('after generator backward')
            

            loss_recon_train.add(reconCriterion(fake_gt_img, image_input_tensors))
            loss_avg_gen.add(disGenCost)
            loss_avg_ocr_unsup.add(opt.ocrWeight * ocrCost)
            loss_avg_style_scode.add(opt.gamma_g * uCost)
            loss_avg_img_recon.add(opt.reconWeight * imgReconCost)
            loss_avg_cycle_recon.add(opt.cycleReconWeight * cycleReconCost)
            loss_avg_vgg_per.add(opt.vggPerWeight * vggPerCost)
            loss_avg_vgg_sty.add(opt.vggStyWeight * vggStyCost)
            loss_avg_vgg_emb.add(opt.vggEmbWeight * vggEmbCost)
            
            if opt.grad_clip !=0.0:
                torch.nn.utils.clip_grad_norm_(genModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
                torch.nn.utils.clip_grad_norm_(cEncoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
                torch.nn.utils.clip_grad_norm_(styleModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)

            optimizer.step()
            
            if not opt.zAlone:
                if get_rank() == 0 and ((iteration + 1) % opt.valInterval == 0 or iteration == 0):
                    #print training images
                    os.makedirs(os.path.join(opt.trainDir,str(iteration)), exist_ok=True)
                    
                    text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)
                    length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(device)

                    #run OCR prediction for gen image (content: gt from input_image_tensors)
                    if 'CTC' in opt.Prediction:
                        _, preds_index = preds_recon.max(2)
                        preds_str_fake_gt_img = converter.decode(preds_index.data, preds_size.data)
                    else:
                        
                        _, preds_index = preds_recon.max(2)
                        preds_str_fake_gt_img = converter.decode(preds_index, length_for_pred)
                        for idx, pred in enumerate(preds_str_fake_gt_img):
                            pred_EOS = pred.find('[s]')
                            preds_str_fake_gt_img[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

                    #render target style image using paired content
                    g_ema.eval()
                    with torch.no_grad():
                        fake_sty_img, _ = g_ema(style, z_c_code, input_is_latent=opt.input_latent)

                    if 'CTC' in opt.Prediction:
                        preds = ocrModel(fake_sty_img, text_z_c, is_train=False, inAct = opt.taskActivation)
                        preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
                        _, preds_index = preds.max(2)
                        preds_str_fake_sty_img = converter.decode(preds_index.data, preds_size.data)
                    else:
                        
                        preds = ocrModel(fake_sty_img, text_z_c[:, :-1], is_train=False, inAct = opt.taskActivation)  # align with Attention.forward
                        _, preds_index = preds.max(2)
                        preds_str_fake_sty_img = converter.decode(preds_index, length_for_pred)
                        for idx, pred in enumerate(preds_str_fake_sty_img):
                            pred_EOS = pred.find('[s]')
                            preds_str_fake_sty_img[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    
                    if opt.cycleReconWeight>0.0:
                        if 'CTC' in opt.Prediction:
                            preds = ocrModel(fake_recon_img, text_gt, is_train=False, inAct = opt.taskActivation)
                            preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
                            _, preds_index = preds.max(2)
                            preds_str_fake_recon_img = converter.decode(preds_index.data, preds_size.data)
                        else:
                            
                            preds = ocrModel(fake_recon_img, text_gt[:, :-1], is_train=False, inAct = opt.taskActivation)  # align with Attention.forward
                            _, preds_index = preds.max(2)
                            preds_str_fake_recon_img = converter.decode(preds_index, length_for_pred)
                            for idx, pred in enumerate(preds_str_fake_recon_img):
                                pred_EOS = pred.find('[s]')
                                preds_str_fake_recon_img[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    
                    for trImgCntr in range(opt.batch_size):
                        try:
                            if not opt.zAlone:
                                utils.save_image(image_output_tensors[trImgCntr],os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_tr_z_orig_'+labels_z_c[trImgCntr]+'_ocr:None'+'.png'),nrow=1,normalize=True,range=(-1, 1))
                            if opt.cEncode == 'cnn':
                                utils.save_image(synth_z_c[trImgCntr],os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_tr_synth_'+labels_z_c[trImgCntr]+'_ocr:None'+'.png'),nrow=1,normalize=True,range=(-1, 1))
                            utils.save_image(fake_gt_img[trImgCntr],os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_tr_gt_recon_pred_'+labels_gt[trImgCntr]+'_ocr:'+preds_str_fake_gt_img[trImgCntr]+'_sty:'+labels_gt[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                            # if opt.reconWeight>0.0:
                            utils.save_image(image_input_tensors[trImgCntr],os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_tr_gt_orig_'+labels_gt[trImgCntr]+'_ocr:None'+'.png'),nrow=1,normalize=True,range=(-1, 1))
                            utils.save_image(fake_sty_img[trImgCntr],os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_tr_z_pred_'+labels_z_c[trImgCntr]+'_ocr:'+preds_str_fake_sty_img[trImgCntr]+'_sty:'+labels_gt[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                            if opt.cycleReconWeight>0.0:
                                utils.save_image(fake_recon_img[trImgCntr],os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_tr_gt_cycle_pred_'+labels_gt[trImgCntr]+'_ocr:'+preds_str_fake_recon_img[trImgCntr]+'_sty:'+labels_gt[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                        except:
                            print('Warning while saving training image')
                            
            g_regularize = cntr % opt.g_reg_every == 0
            # print('before g_regularize')
            
            if g_regularize:
                path_batch_size = max(1, opt.batch_size // opt.path_batch_shrink)

                image_input_tensors, _, labels_gt, labels_z_c, labelSynthImg, synth_z_c  = next(train_loader)
                # labels_z_c, synth_z_c = next(text_loader)
                # print(labels_z_c)

                # image_input_tensors = image_input_tensors.to(device)
                image_input_tensors = image_input_tensors[:path_batch_size].to(device)
                # gt_image_tensors = image_input_tensors[:path_batch_size].detach()    #exemplar word style image; training OCR
                synth_z_c = synth_z_c[:path_batch_size].to(device)
                labelSynthImg = labelSynthImg[:path_batch_size].to(device)

                text_gt, length_gt = converter.encode(labels_gt[:path_batch_size], batch_max_length=opt.batch_max_length)
                
                if opt.zAlone:
                    z_gt_code = None
                    style = None
                else:
                    if opt.cEncode == 'mlp':    
                        z_gt_code = cEncoder(text_gt)
                    elif opt.cEncode == 'cnn':    
                        z_gt_code = cEncoder(labelSynthImg)
                    # print('after g_regularize cEncoder')
                    
                    style = styleModel(image_input_tensors)
                    # print('after g_regularize styleModel')
                    
                
                # if opt.noiseConcat or opt.zAlone:
                #     style = mixing_noise(path_batch_size, opt.latent, opt.mixing, device, style)
                # else:
                #     style = [style]
                

                fake_gt_img, latents = genModel(style, z_gt_code, return_latents=True, input_is_latent=opt.input_latent)
                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    fake_gt_img, latents, mean_path_length
                )
                # print('after g_regularize genModel')
                
                genModel.zero_grad()
                if not opt.zAlone:
                    cEncoder.zero_grad()
                    styleModel.zero_grad()
                weighted_path_loss = opt.path_regularize * opt.g_reg_every * path_loss

                if opt.path_batch_shrink:
                    weighted_path_loss += 0 * fake_gt_img[0, 0, 0, 0]

                weighted_path_loss.backward()
                # print('after g_regularize backward')
                
                if opt.grad_clip !=0.0:
                    torch.nn.utils.clip_grad_norm_(genModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
                    torch.nn.utils.clip_grad_norm_(cEncoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
                    torch.nn.utils.clip_grad_norm_(styleModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)

                optimizer.step()

                mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
                )

            accumulate(g_ema, genModel_module, accum)

            log_avg_path_loss_val.add(path_loss)
            log_avg_mean_path_length_avg.add(torch.tensor(mean_path_length_avg))
            log_ada_aug_p.add(torch.tensor(ada_aug_p))
            # print('after g_regularize')
            

        if get_rank() == 0 or opt.testFlag:
            if wandb and opt.wandb:
                wandb.log(
                    {
                        "Generator": loss_avg_gen.val().item(),
                        "Discriminator": loss_avg_dis.val().item(),
                        "Train-UnSup-OCR-Loss": loss_avg_ocr_unsup.val().item(),
                        "Train-ImageRecon-Loss": loss_avg_img_recon.val().item(),
                        "Train-CycleRecon-Loss": loss_avg_cycle_recon.val().item(),
                        "Train-VGGPer-Loss": loss_avg_vgg_per.val().item(),
                        "Train-VGGSty-Loss": loss_avg_vgg_sty.val().item(),
                        "Train-VGGEmb-Loss": loss_avg_vgg_emb.val().item(),
                        "Train-r1_val": log_r1_val.val().item(),
                        "Train-path_loss_val": log_avg_path_loss_val.val().item(),
                        "Train-mean_path_length_avg": log_avg_mean_path_length_avg.val().item(),
                        "Train-StyleImgRecon-Loss')": loss_recon_train.val().item(),
                        "Val-StyleImgRecon-Loss": loss_recon_val.val().item()
                    }
                )
        
            # print
            if (iteration + 1) % opt.valInterval == 0 or iteration == 0 or opt.testFlag: # To see training progress, we also conduct validation when 'iteration == 0' 
                #validation
                iCntr=0
                evalCntr=0
                
                valMSE=0.0
                valSSIM=0.0
                valPSNR=0.0
                c1_s1_input_correct=0.0
                c2_s1_gen_correct=0.0
                c1_s1_input_ed_correct=0.0
                c2_s1_gen_ed_correct=0.0
                


                text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)
                length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(device)
                ims, txts = [], []
                for vCntr, (image_input_tensors, image_output_tensors, labels_gt, labels_z_c, labelSynthImg, synth_z_c) in enumerate(valid_loader):

                    if opt.debugFlag and vCntr >2:
                        break  
                    
                    if not(opt.testFlag) and iCntr>500:
                        break 
                    
                    # if not(opt.testFlag) and vCntr>25:
                    #     break              
                    # labels_gt_1 = labels[:opt.batch_size]
                    # labels_gt_2 = labels[opt.batch_size:]

                    #generate paired content with similar style
                    # labels_z_c_1, synth_z_c_1 = next(text_loader)
                    # labels_z_c_2, synth_z_c_2 = next(text_loader)
                    
                    image_input_tensors = image_input_tensors.to(device)
                    image_output_tensors = image_output_tensors.to(device)
                    # gt_image_tensors_1 = image_input_tensors[:opt.batch_size].detach()    #exemplar word style image; training OCR
                    # gt_image_tensors_2 = image_input_tensors[opt.batch_size:].detach()

                    if opt.realVaData:
                        labels_z_c, synth_z_c = next(text_loader)
                    
                    labelSynthImg = labelSynthImg.to(device)
                    synth_z_c = synth_z_c.to(device)
                    
                    text_z_c, length_z_c = converter.encode(labels_z_c, batch_max_length=opt.batch_max_length)
                    # text_z_c_2, length_z_c_2 = converter.encode(labels_z_c_2, batch_max_length=opt.batch_max_length)
                    text_gt, length_gt = converter.encode(labels_gt, batch_max_length=opt.batch_max_length)
                    # text_gt_2, length_gt_2 = converter.encode(labels_gt_2, batch_max_length=opt.batch_max_length)

                    if not opt.zAlone:
                        cEncoder.eval()
                        styleModel.eval()
                    g_ema.eval()
                    disEncModel.eval()

                    with torch.no_grad():
                        if opt.zAlone:
                            z_c_code = None
                            # z_c_code_2 = None
                            style = None
                            # style_2 = None
                        else:
                            if opt.cEncode == 'mlp':    
                                z_c_code = cEncoder(text_z_c)
                                z_gt_code = cEncoder(text_gt)
                            elif opt.cEncode == 'cnn':    
                                z_c_code = cEncoder(synth_z_c)
                                z_gt_code = cEncoder(labelSynthImg)
                                
                            style = styleModel(image_input_tensors)
                            # style_2 = styleModel(gt_image_tensors_2)
                        
                        # if opt.noiseConcat or opt.zAlone:
                        #     style = mixing_noise(opt.batch_size, opt.latent, opt.mixing, device, style)
                        #     # style_2 = mixing_noise(opt.batch_size, opt.latent, opt.mixing, device, style_2)
                        # else:
                        #     style = [style]
                            # style_2 = [style_2]
                        # print('inside valoidatoin before genModel c1 s1')
                        
                        fake_img_c1_s1, _ = g_ema(style, z_gt_code, input_is_latent=opt.input_latent)
                        # print('inside valoidatoin after genModel c1 s1')
                        
                        if not opt.zAlone:
                            fake_img_c2_s1, _ = g_ema(style, z_c_code, input_is_latent=opt.input_latent)
                            if not(opt.realVaData):
                                loss_recon_val.add(reconCriterion(fake_img_c2_s1, image_output_tensors)) 
                            else:
                                loss_recon_val.add(torch.tensor(0.0))

                        if not opt.zAlone:
                            #Run OCR prediction
                            if 'CTC' in opt.Prediction:
                                
                                preds = ocrModel(fake_img_c1_s1, text_gt, is_train=False, inAct = opt.taskActivation)
                                preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
                                _, preds_index = preds.max(2)
                                preds_str_fake_img_c1_s1 = converter.decode(preds_index.data, preds_size.data)

                                preds = ocrModel(fake_img_c2_s1, text_z_c, is_train=False, inAct = opt.taskActivation)
                                preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
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

                                # print("Not implemented error")
                                # sys.exit()
                        else:
                            # print("Not implemented error")
                            # sys.exit()
                            preds_str_fake_img_c1_s1 = [':None:'] * fake_img_c1_s1.shape[0]
                            # preds_str_gt = [':None:'] * fake_img_c1_s1.shape[0] 


                    if not opt.testFlag:
                        if not opt.zAlone:
                            cEncoder.train()
                            styleModel.train()
                        disEncModel.train()

                    if opt.testFlag:
                        pathPrefix = os.path.join(opt.valDir,str(start_iter))
                    else:
                        pathPrefix = os.path.join(opt.valDir,str(iteration))
        
                    os.makedirs(os.path.join(pathPrefix), exist_ok=True)
                    # if not opt.testFlag:
                    #     webpage = html.HTML(pathPrefix, 'Experiment name = %s' % 'Validation')
                    #     webpage.add_header('Validation iteration [%d]' % iteration)
                    
                    
                    for trImgCntr in range(image_output_tensors.shape[0]):
                        
                        #evaluations
                        valRange = (-1,+1)
                        gtTensor = tensor2im(image_output_tensors[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1]))
                        predTensor = tensor2im(fake_img_c2_s1[trImgCntr].clone().clamp_(min=valRange[0], max=valRange[1]))
                        
                        if not(opt.realVaData):
                            evalMSE = mean_squared_error(gtTensor/255, predTensor/255)
                            evalSSIM = structural_similarity(gtTensor, predTensor, data_range=predTensor.max() - predTensor.min(), multichannel=True)
                            evalPSNR = peak_signal_noise_ratio(gtTensor, predTensor, data_range=predTensor.max() - predTensor.min())

                            valMSE+=evalMSE
                            valSSIM+=evalSSIM
                            valPSNR+=evalPSNR

                        #ocr accuracy
                        # for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
                        c1_s1_input_gt = labels_gt[trImgCntr]
                        c1_s1_input_ocr = preds_str_gt_1[trImgCntr]
                        c2_s1_gen_gt = labels_z_c[trImgCntr]
                        c2_s1_gen_ocr = preds_str_fake_img_c2_s1[trImgCntr]


                        # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                        if opt.sensitive and opt.data_filtering_off:
                            c1_s1_input_gt = c1_s1_input_gt.lower()
                            c1_s1_input_ocr = c1_s1_input_ocr.lower()
                            c2_s1_gen_gt = c2_s1_gen_gt.lower()
                            c2_s1_gen_ocr = c2_s1_gen_ocr.lower()

                            alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                            out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                            c1_s1_input_gt = re.sub(out_of_alphanumeric_case_insensitve, '', c1_s1_input_gt)
                            c1_s1_input_ocr = re.sub(out_of_alphanumeric_case_insensitve, '', c1_s1_input_ocr)
                            c2_s1_gen_gt = re.sub(out_of_alphanumeric_case_insensitve, '', c2_s1_gen_gt)
                            c2_s1_gen_ocr = re.sub(out_of_alphanumeric_case_insensitve, '', c2_s1_gen_ocr)

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

                                # if opt.testFlag:
                                if iCntr == 0:
                                    # update website
                                    webpage = html.HTML(pathPrefix, 'Experiment name = %s' % 'Test')
                                    if opt.testFlag:
                                        webpage.add_header('Testing iteration [%d]' % iteration)
                                    else:
                                        webpage.add_header('Validation iteration [%d]' % iteration)
                                
                                iCntr += 1
                                # else:
                                #     iCntr = trImgCntr 
                                
                                img_path_c1_s1 = os.path.join(str(iCntr)+'_pred_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr:'+preds_str_fake_img_c1_s1[trImgCntr]+'.png')
                                img_path_gt_1 = os.path.join(str(iCntr)+'_gt_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr:'+preds_str_gt_1[trImgCntr]+'.png')
                                img_path_gt_2 = os.path.join(str(iCntr)+'_gt_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr:'+preds_str_gt_2[trImgCntr]+'.png')
                                img_path_c2_s1 = os.path.join(str(iCntr)+'_pred_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr:'+preds_str_fake_img_c2_s1[trImgCntr]+'.png')
                                
                                ims.append([img_path_gt_1, img_path_c1_s1, img_path_gt_2, img_path_c2_s1])

                                content_c1_s1 = 'PSTYLE-1 '+'Text-1:' + labels_gt[trImgCntr]+' OCR:' + preds_str_fake_img_c1_s1[trImgCntr]
                                content_gt_1 = 'OSTYLE-1 '+'GT:' + labels_gt[trImgCntr]+' OCR:' + preds_str_gt_1[trImgCntr]
                                content_gt_2 = 'OSTYLE-1 '+'GT:' + labels_z_c[trImgCntr]+' OCR:'+preds_str_gt_2[trImgCntr]
                                content_c2_s1 = 'PSTYLE-1 '+'Text-2:' + labels_z_c[trImgCntr]+' OCR:'+preds_str_fake_img_c2_s1[trImgCntr]
                                
                                txts.append([content_gt_1, content_c1_s1, content_gt_2, content_c2_s1])
                                
                                utils.save_image(fake_img_c1_s1[trImgCntr],os.path.join(pathPrefix,str(iCntr)+'_pred_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr:'+preds_str_fake_img_c1_s1[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                                
                                if not opt.zAlone:
                                    utils.save_image(image_input_tensors[trImgCntr],os.path.join(pathPrefix,str(iCntr)+'_gt_val_c1_s1_'+labels_gt[trImgCntr]+'_ocr:'+preds_str_gt_1[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                                    utils.save_image(image_output_tensors[trImgCntr],os.path.join(pathPrefix,str(iCntr)+'_gt_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr:'+preds_str_gt_2[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                                    utils.save_image(fake_img_c2_s1[trImgCntr],os.path.join(pathPrefix,str(iCntr)+'_pred_val_c2_s1_'+labels_z_c[trImgCntr]+'_ocr:'+preds_str_fake_img_c2_s1[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                                    
                                
                            except:
                                print('Warning while saving validation image')
                
                webpage.add_images(ims, txts, width=256, realFlag=opt.realVaData)    
                elapsed_time = time.time() - start_time
                webpage.save()
                
                avg_valMSE = valMSE/float(evalCntr)
                avg_valSSIM = valSSIM/float(evalCntr)
                avg_valPSNR = valPSNR/float(evalCntr)
                avg_c1_s1_input_wer = c1_s1_input_correct/float(evalCntr)
                avg_c2_s1_gen_wer = c2_s1_gen_correct/float(evalCntr)
                avg_c1_s1_input_cer = c1_s1_input_ed_correct/float(evalCntr)
                avg_c2_s1_gen_cer = c2_s1_gen_ed_correct/float(evalCntr)

                if not(opt.realVaData):
                    if avg_valMSE < bestModelError:
                        bestModelError = avg_valMSE
                        with open(os.path.join(opt.exp_dir,opt.exp_name,'log_train_best.txt'), 'a') as log:
                            loss_log = f'[{iteration+1}/{opt.num_iter}]  \
                                Test MSE: {avg_valMSE:0.5f}, Test SSIM: {avg_valSSIM:0.5f}, , Test PSNR: {avg_valPSNR:0.5f}, \
                                Test Input Word Acc: {avg_c1_s1_input_wer:0.5f}, Test Gen Word Acc: {avg_c2_s1_gen_wer:0.5f}, \
                                Test Input Char Acc: {avg_c1_s1_input_cer:0.5f}, Test Gen Char Acc: {avg_c2_s1_gen_cer:0.5f}, \
                                Elapsed_time: {elapsed_time:0.5f}'
                            
                            print(loss_log)
                            log.write(loss_log+"\n")
                            
                            #save best model
                            if opt.zAlone:
                                torch.save({
                                'genModel':genModel_module.state_dict(),
                                'g_ema':g_ema.state_dict(),
                                'disEncModel':disEncModel_module.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'dis_optimizer':dis_optimizer.state_dict(),
                                'bestModelError':bestModelError}, 
                                os.path.join(opt.exp_dir,opt.exp_name,'bestvalmodel_synth.pth'))
                            else:
                                torch.save({
                                'cEncoder':cEncoder_module.state_dict(),
                                'styleModel':styleModel_module.state_dict(),
                                'genModel':genModel_module.state_dict(),
                                'g_ema':g_ema.state_dict(),
                                'ocrModel':ocrModel_module.state_dict(),
                                'disEncModel':disEncModel_module.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'ocr_optimizer':ocr_optimizer.state_dict(),
                                'dis_optimizer':dis_optimizer.state_dict(),
                                'bestModelError':bestModelError}, 
                                os.path.join(opt.exp_dir,opt.exp_name,'bestvalmodel_synth.pth'))

                if not opt.testFlag:
                    #DO HERE
                    with open(os.path.join(opt.exp_dir,opt.exp_name,'log_train.txt'), 'a') as log:

                        # training loss and validation loss
                        
                        loss_log = f'[{iteration+1}/{opt.num_iter}]  \
                            Train Dis loss: {loss_avg_dis.val():0.5f}, Train Gen loss: {loss_avg_gen.val():0.5f},\
                            Train UnSup OCR loss: {loss_avg_ocr_unsup.val():0.5f}, \
                            Train Image Recon loss: {loss_avg_img_recon.val():0.5f}, \
                            Train Cycle Recon loss: {loss_avg_cycle_recon.val():0.5f}, \
                            Train VGG Per loss: {loss_avg_vgg_per.val():0.5f}, Train VGG Style loss: {loss_avg_vgg_sty.val():0.5f}, \
                            Train VGG Embed loss: {loss_avg_vgg_emb.val():0.5f}, \
                            Train R1-val loss: {log_r1_val.val():0.5f}, Train avg-path-loss: {log_avg_path_loss_val.val():0.5f}, \
                            Train mean-path-length loss: {log_avg_mean_path_length_avg.val():0.5f}, \
                            Train StyleImgRecon loss: {loss_recon_train.val():0.5f}, Val StyleImgRecon loss: {loss_recon_val.val():0.5f}, \
                            Val MSE: {avg_valMSE:0.5f}, Val SSIM: {avg_valSSIM:0.5f}, , Val PSNR: {avg_valPSNR:0.5f}, \
                            Val Input Word Acc: {avg_c1_s1_input_wer:0.5f}, Val Gen Word Acc: {avg_c2_s1_gen_wer:0.5f}, \
                            Val Input Char Acc: {avg_c1_s1_input_cer:0.5f}, Val Gen Char Acc: {avg_c2_s1_gen_cer:0.5f}, \
                            Elapsed_time: {elapsed_time:0.5f}'
                        
                        
                        #plotting
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-Dis-Loss'), loss_avg_dis.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-Gen-Loss'), loss_avg_gen.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-UnSup-OCR-Loss'), loss_avg_ocr_unsup.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-ImageRecon-Loss'), loss_avg_img_recon.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-CycleRecon-Loss'), loss_avg_cycle_recon.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-VGGPer-Loss'), loss_avg_vgg_per.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-VGGSty-Loss'), loss_avg_vgg_sty.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-VGGEmb-Loss'), loss_avg_vgg_emb.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-r1_val'), log_r1_val.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-path_loss_val'), log_avg_path_loss_val.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-mean_path_length_avg'), log_avg_mean_path_length_avg.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-StyleImgRecon-Loss'), loss_recon_train.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Val-StyleImgRecon-Loss'), loss_recon_val.val().item())

                        lib.plot.plot(os.path.join(opt.plotDir,'Val-MSE'), avg_valMSE)
                        lib.plot.plot(os.path.join(opt.plotDir,'Val-PSNR'), avg_valPSNR)
                        lib.plot.plot(os.path.join(opt.plotDir,'Val-SSIM'), avg_valSSIM)

                        lib.plot.plot(os.path.join(opt.plotDir,'Val-Input-Word-Acc'), avg_c1_s1_input_wer)
                        lib.plot.plot(os.path.join(opt.plotDir,'Val-Gen-Word-Acc'), avg_c2_s1_gen_wer)
                        lib.plot.plot(os.path.join(opt.plotDir,'Val-Input-Char-Acc'), avg_c1_s1_input_cer)
                        lib.plot.plot(os.path.join(opt.plotDir,'Val-Gen-Char-Acc'), avg_c2_s1_gen_cer)

                        
                        print(loss_log)
                        log.write(loss_log+"\n")

                        loss_recon_train.reset()
                        loss_recon_val.reset()
                        loss_avg_dis.reset()
                        loss_avg_gen.reset()
                        loss_avg_ocr_unsup.reset()
                        loss_avg_ocr_sup.reset()
                        loss_avg_style_ucode.reset()
                        loss_avg_style_scode.reset()
                        loss_avg_img_recon.reset()
                        loss_avg_cycle_recon.reset()
                        loss_avg_vgg_per.reset()
                        loss_avg_vgg_sty.reset()
                        loss_avg_vgg_emb.reset()
                        log_r1_val.reset()
                        log_avg_path_loss_val.reset()
                        log_avg_mean_path_length_avg.reset()
                        log_ada_aug_p.reset()
                        

                    lib.plot.flush()
                else:
                    with open(os.path.join(opt.exp_dir,opt.exp_name,'log_test.txt'), 'a') as log:

                        # training loss and validation loss
                        
                        loss_log = f'[{iteration+1}/{opt.num_iter}]  \
                            Test MSE: {avg_valMSE:0.5f}, Test SSIM: {avg_valSSIM:0.5f}, , Test PSNR: {avg_valPSNR:0.5f}, \
                            Test Input Word Acc: {avg_c1_s1_input_wer:0.5f}, Test Gen Word Acc: {avg_c2_s1_gen_wer:0.5f}, \
                            Test Input Char Acc: {avg_c1_s1_input_cer:0.5f}, Test Gen Char Acc: {avg_c2_s1_gen_cer:0.5f}, \
                            Elapsed_time: {elapsed_time:0.5f}'
                        
                        print(loss_log)
                        log.write(loss_log+"\n")
                

            lib.plot.tick()
            
            if not opt.testFlag:
                # save model per 30000 iter.
                if (iteration) % 15000 == 0:
                    if opt.zAlone:
                        torch.save({
                        'genModel':genModel_module.state_dict(),
                        'g_ema':g_ema.state_dict(),
                        'disEncModel':disEncModel_module.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'dis_optimizer':dis_optimizer.state_dict(),
                        'bestModelError':bestModelError}, 
                        os.path.join(opt.exp_dir,opt.exp_name,'iter_'+str(iteration+1)+'_synth.pth'))
                    else:
                        torch.save({
                        'cEncoder':cEncoder_module.state_dict(),
                        'styleModel':styleModel_module.state_dict(),
                        'genModel':genModel_module.state_dict(),
                        'g_ema':g_ema.state_dict(),
                        'ocrModel':ocrModel_module.state_dict(),
                        'disEncModel':disEncModel_module.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'ocr_optimizer':ocr_optimizer.state_dict(),
                        'dis_optimizer':dis_optimizer.state_dict(),
                        'bestModelError':bestModelError}, 
                        os.path.join(opt.exp_dir,opt.exp_name,'iter_'+str(iteration+1)+'_synth.pth'))
        # print('outside validation')
        
        if opt.testFlag:
            print('end the testing')
            sys.exit()

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1
        cntr+=1


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'None':
        scheduler = None # constant scheduler
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                                        gamma=opt.gamma, last_epoch=-1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    
    return scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp18/', help='Where to store logs and models')
    parser.add_argument('--exp_name', default='debug', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--words_file', required=True, default='', help="path to words file. phoc will be sampled from this file")
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=900000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=3000, help='Interval between each validation')
    parser.add_argument('--saved_ocr_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_synth_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_gen_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_font_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--optim', default='adam', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--lr_policy', default='None', help='None|step')
    parser.add_argument('--step_size', type=int, default=100000, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='how much to decay learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping value. default=5')
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
    parser.add_argument('--ocrFixed', action='store_true', help='true: for pretrined OCR and fixed weights')
    parser.add_argument('--ocrWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--reconWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--cycleReconWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--disWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--gamma_g', type=float, default=0.0, help='weights for loss')
    parser.add_argument('--gamma_e', type=float, default=0.0, help='weights for loss')
    parser.add_argument('--beta', type=float, default=0.0, help='weights for loss')
    parser.add_argument('--alpha', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--vggPerWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--vggStyWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--vggEmbWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--vggNoMean', action='store_true', help='if yes; No mean normalization is applied to vgg input')
    parser.add_argument('--styleDetach', action='store_true', help='whether to detach style')
    parser.add_argument('--gan_type', default='lsgan', help='lsgan/nsgan/wgan')
    parser.add_argument('--imgReconLoss', default='l1', help='l1/l2')
    parser.add_argument('--styleLoss', default='l1', help='l1/triplet')
    parser.add_argument('--contentLoss', default='pred', help='pred(ctc/attn)/vis/seq')
    parser.add_argument('--tripletMargin', type=float, default=1.0, help='triplet margin')
    parser.add_argument('--style_input', action='store_true', help='whether target style input is given for training/validation')
    parser.add_argument('--style_content_input', action='store_true', help='whether target  input content image is given for training/validation')
    parser.add_argument('--styleNorm', default='bn', help='bn/in')
    parser.add_argument('--contentNorm', default='bn', help='bn/in')

    parser.add_argument('--debugFlag', action='store_true', help='for debugging')
    parser.add_argument('--modelFolderFlag', action='store_true', help='load latest files from saved model folder')
    parser.add_argument('--testFlag', action='store_true', help='for testing')
    parser.add_argument('--visFlag', action='store_true', help='for visualization')

    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    # parser.add_argument("--path_batch_shrink", type=int, default=1)
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
    parser.add_argument("--style_latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--input_latent", action="store_true")
    parser.add_argument("--content_inject_index", type=int, default=1)
    parser.add_argument("--zAlone", action="store_true", help="test original style GAN")
    parser.add_argument("--noiseConcat", action="store_true", help="noise concat with style")

    parser.add_argument("--grad_balance", action="store_true", help="use gradient balancing")
    parser.add_argument("--grad_all_balance", action="store_true", help="use gradient all balancing")
    parser.add_argument('--cEncode', default='mlp', help='mlp/cnn')
    parser.add_argument('--taskActivation', default=None, help='None/tanh')
    parser.add_argument("--numLatents", type=int, default=1)
    parser.add_argument("--realTrData", action="store_true", help="flag for training real data where we don't have target style GT")
    parser.add_argument("--realVaData", action="store_true", help="flag for validation/testing real data where we don't have target style GT")

    opt = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.distributed = n_gpu > 1

    if opt.distributed:
        print("Running distributed setting: Num gpus::", n_gpu)
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    if opt.zAlone:
        opt.gamma_e = 0.0
        opt.gamma_g = 0.0
        opt.beta = 0.0
        opt.reconWeight = 0.0
        opt.vggPerWeight = 0.0
        opt.vggStyWeight = 0.0

    if opt.rgb:
        opt.input_channel = 3
    
    if opt.taskActivation == 'None':
        opt.taskActivation = None

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name), exist_ok=True)
    opt.trainDir=os.path.join(opt.exp_dir,opt.exp_name,'trainImages')
    if opt.testFlag:
        opt.valDir=os.path.join(opt.exp_dir,opt.exp_name,'testImages')    
    else:
        opt.valDir=os.path.join(opt.exp_dir,opt.exp_name,'valImages')
    opt.plotDir=os.path.join(opt.exp_dir,opt.exp_name,'plots')
    
    os.makedirs(opt.trainDir, exist_ok=True)
    os.makedirs(opt.valDir, exist_ok=True)
    os.makedirs(opt.plotDir, exist_ok=True)


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
    # opt.num_gpu = torch.cuda.device_count()

    if get_rank() == 0 and wandb is not None and opt.wandb:
        wandb.init(project="Scribe")

    train(opt)

