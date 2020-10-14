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
from dataset import hierarchical_dataset, AlignPairCollate, Batch_Balanced_Dataset, tensor2im, save_image, phoc_gen, text_gen, LmdbStyleDataset
# from model import ModelV1, StyleTensorEncoder, StyleLatentEncoder, MsImageDisV2, AdaIN_Tensor_WordGenerator, VGGPerceptualLossModel, Mixer
from model import ModelV1, GlobalContentEncoder
from test_synth import validation_synth_v7

import tflib as lib
import tflib.plot

sys.path.append('/private/home/pkrishnan/codes/st-scribe/stylegan2-pytorch/')

try:
    import wandb

except ImportError:
    wandb = None

from model_word import GeneratorV4 as styleGANGen 
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


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


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
        return torch.cat((torch.randn(batch, latent_dim, device=device),z_code),1)

    noises = torch.cat((torch.randn(n_noise, batch, latent_dim, device=device),z_code.repeat(2,1,1)),2).unbind(0)

    return noises


def mixing_noise(z_code, batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(z_code, batch, latent_dim, 2, device)

    else:
        return [make_noise(z_code, batch, latent_dim, 1, device)]


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
    AlignCollate_valid = AlignPairCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    train_dataset = LmdbStyleDataset(root=opt.train_data, opt=opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size*2, #*2 to sample different images from training encoder and discriminator real images
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True, drop_last=True)
    
    print('-' * 80)
    
    valid_dataset = LmdbStyleDataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size*2, #*2 to sample different images from training encoder and discriminator real images
        shuffle=False,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True, drop_last=True)
    
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    text_dataset = text_gen(opt)
    text_loader = torch.utils.data.DataLoader(
        text_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        pin_memory=True, drop_last=True)
    opt.num_class = len(converter.character)
    

    c_code_size = opt.latent
    cEncoder = GlobalContentEncoder(opt.num_class, text_len, opt.char_embed_size, c_code_size)
    ocrModel = ModelV1(opt)

    if opt.zAlone:
        genModel = styleGANGen(opt.size, opt.latent, opt.latent, opt.n_mlp, channel_multiplier=opt.channel_multiplier)
        g_ema = styleGANGen(opt.size, opt.latent, opt.latent, opt.n_mlp, channel_multiplier=opt.channel_multiplier)
    else:
        genModel = styleGANGen(opt.size, opt.latent+c_code_size, opt.latent, opt.n_mlp, channel_multiplier=opt.channel_multiplier)
        g_ema = styleGANGen(opt.size, opt.latent+c_code_size, opt.latent, opt.n_mlp, channel_multiplier=opt.channel_multiplier)
    disEncModel = styleGANDis(opt.size, channel_multiplier=opt.channel_multiplier, input_dim=opt.input_channel, code_s_dim=c_code_size)
    
    accumulate(g_ema, genModel, 0)
    
    # uCriterion = torch.nn.MSELoss()
    # sCriterion = torch.nn.MSELoss()
    # if opt.contentLoss == 'vis' or opt.contentLoss == 'seq':
    #     ocrCriterion = torch.nn.L1Loss()
    # else:
    if 'CTC' in opt.Prediction:
        ocrCriterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        print('Not implemented error')
        sys.exit()
        # ocrCriterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    cEncoder= torch.nn.DataParallel(cEncoder).to(device)
    cEncoder.train()
    genModel = torch.nn.DataParallel(genModel).to(device)
    g_ema = torch.nn.DataParallel(g_ema).to(device)
    genModel.train()
    g_ema.eval()

    disEncModel = torch.nn.DataParallel(disEncModel).to(device)
    disEncModel.train()

    ocrModel = torch.nn.DataParallel(ocrModel).to(device)
    if opt.ocrFixed:
        if opt.Transformation == 'TPS':
            ocrModel.module.Transformation.eval()
        ocrModel.module.FeatureExtraction.eval()
        ocrModel.module.AdaptiveAvgPool.eval()
        # ocrModel.module.SequenceModeling.eval()
        ocrModel.module.Prediction.eval()
    else:
        ocrModel.train()

    g_reg_ratio = opt.g_reg_every / (opt.g_reg_every + 1)
    d_reg_ratio = opt.d_reg_every / (opt.d_reg_every + 1)

    
    optimizer = optim.Adam(
        list(genModel.parameters())+list(cEncoder.parameters()),
        lr=opt.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    dis_optimizer = optim.Adam(
        disEncModel.parameters(),
        lr=opt.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    
    ocr_optimizer = optim.Adam(
        ocrModel.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.99),
    )


    ## Loading pre-trained files
    if opt.modelFolderFlag:
        if len(glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth")))>0:
            opt.saved_synth_model = glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth"))[-1]

    if opt.saved_ocr_model !='' and opt.saved_ocr_model !='None':
        print(f'loading pretrained ocr model from {opt.saved_ocr_model}')
        checkpoint = torch.load(opt.saved_ocr_model)
        ocrModel.load_state_dict(checkpoint)
    
    # if opt.saved_gen_model !='' and opt.saved_gen_model !='None':
    #     print(f'loading pretrained gen model from {opt.saved_gen_model}')
    #     checkpoint = torch.load(opt.saved_gen_model, map_location=lambda storage, loc: storage)
    #     genModel.module.load_state_dict(checkpoint['g'])
    #     g_ema.module.load_state_dict(checkpoint['g_ema'])

    if opt.saved_synth_model != '' and opt.saved_synth_model != 'None':
        print(f'loading pretrained synth model from {opt.saved_synth_model}')
        checkpoint = torch.load(opt.saved_synth_model)
        
        # styleModel.load_state_dict(checkpoint['styleModel'])
        # mixModel.load_state_dict(checkpoint['mixModel'])
        genModel.load_state_dict(checkpoint['genModel'])
        g_ema.load_state_dict(checkpoint['g_ema'])
        disEncModel.load_state_dict(checkpoint['disEncModel'])
        ocrModel.load_state_dict(checkpoint['ocrModel'])
        
        optimizer.load_state_dict(checkpoint["optimizer"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        ocr_optimizer.load_state_dict(checkpoint["ocr_optimizer"])

    # if opt.imgReconLoss == 'l1':
    #     recCriterion = torch.nn.L1Loss()
    # elif opt.imgReconLoss == 'ssim':
    #     recCriterion = ssim
    # elif opt.imgReconLoss == 'ms-ssim':
    #     recCriterion = msssim
    

    # loss averager
    loss_avg_dis = Averager()
    loss_avg_gen = Averager()
    loss_avg_unsup = Averager()
    loss_avg_sup = Averager()
    log_r1_val = Averager()
    log_avg_path_loss_val = Averager()
    log_avg_mean_path_length_avg = Averager()
    log_ada_aug_p = Averager()
    loss_avg_ocr_sup = Averager()
    loss_avg_ocr_unsup = Averager()

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
    ocr_scheduler = get_scheduler(ocr_optimizer,opt)

    start_time = time.time()
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

    # sample_z = torch.randn(opt.n_sample, opt.latent, device=device)

    while(True):
        # print(cntr)
        # train part
        if opt.lr_policy !="None":
            scheduler.step()
            dis_scheduler.step()
            ocr_scheduler.step()
        
        image_input_tensors, _, labels, _ = iter(train_loader).next()
        labels_z_c = iter(text_loader).next()

        image_input_tensors = image_input_tensors.to(device)
        gt_image_tensors = image_input_tensors[:opt.batch_size].detach()
        real_image_tensors = image_input_tensors[opt.batch_size:].detach()
        
        labels_gt = labels[:opt.batch_size]
        
        requires_grad(cEncoder, False)
        requires_grad(genModel, False)
        requires_grad(disEncModel, True)
        requires_grad(ocrModel, False)

        text_z_c, length_z_c = converter.encode(labels_z_c, batch_max_length=opt.batch_max_length)
        text_gt, length_gt = converter.encode(labels_gt, batch_max_length=opt.batch_max_length)

        z_c_code = cEncoder(text_z_c)
        style = mixing_noise(z_c_code, opt.batch_size, opt.latent, opt.mixing, device)
        
        if opt.zAlone:
            #to validate orig style gan results
            newstyle = []
            newstyle.append(style[0][:,:opt.latent])
            if len(style)>1:
                newstyle.append(style[1][:,:opt.latent])
            style = newstyle
        
        fake_img,_ = genModel(style, input_is_latent=opt.input_latent)
        
        # #unsupervised code prediction on generated image
        # u_pred_code = disEncModel(fake_img, mode='enc')
        # uCost = uCriterion(u_pred_code, z_code)

        # #supervised code prediction on gt image
        # s_pred_code = disEncModel(gt_image_tensors, mode='enc')
        # sCost = uCriterion(s_pred_code, gt_phoc_tensors)

        #Domain discriminator
        fake_pred = disEncModel(fake_img)
        real_pred = disEncModel(real_image_tensors)
        disCost = d_logistic_loss(real_pred, fake_pred)

        # dis_cost = disCost + opt.gamma_e*uCost + opt.beta*sCost

        loss_avg_dis.add(disCost)
        # loss_avg_sup.add(opt.beta*sCost)
        # loss_avg_unsup.add(opt.gamma_e * uCost)

        disEncModel.zero_grad()
        disCost.backward()
        dis_optimizer.step()

        d_regularize = cntr % opt.d_reg_every == 0

        if d_regularize:
            real_image_tensors.requires_grad = True
            real_pred = disEncModel(real_image_tensors)
            
            r1_loss = d_r1_loss(real_pred, real_image_tensors)

            disEncModel.zero_grad()
            (opt.r1 / 2 * r1_loss * opt.d_reg_every + 0 * real_pred[0]).backward()

            dis_optimizer.step()
        log_r1_val.add(r1_loss)
        
        # Recognizer update
        if not opt.ocrFixed and not opt.zAlone:
            requires_grad(disEncModel, False)
            requires_grad(ocrModel, True)

            if 'CTC' in opt.Prediction:
                preds_recon = ocrModel(gt_image_tensors, text_gt, is_train=True)
                preds_size = torch.IntTensor([preds_recon.size(1)] * opt.batch_size)
                preds_recon_softmax = preds_recon.log_softmax(2).permute(1, 0, 2)
                ocrCost = ocrCriterion(preds_recon_softmax, text_gt, preds_size, length_gt)
            else:
                print("Not implemented error")
                sys.exit()
            
            ocrModel.zero_grad()
            ocrCost.backward()
            # torch.nn.utils.clip_grad_norm_(ocrModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            ocr_optimizer.step()
            loss_avg_ocr_sup.add(ocrCost)
        else:
            loss_avg_ocr_sup.add(torch.tensor(0.0))


        # [Word Generator] update
        # image_input_tensors, _, labels, _ = iter(train_loader).next()
        labels_z_c = iter(text_loader).next()

        # image_input_tensors = image_input_tensors.to(device)
        # gt_image_tensors = image_input_tensors[:opt.batch_size]
        # real_image_tensors = image_input_tensors[opt.batch_size:]
        
        # labels_gt = labels[:opt.batch_size]

        requires_grad(cEncoder, True)
        requires_grad(genModel, True)
        requires_grad(disEncModel, False)
        requires_grad(ocrModel, False)

        text_z_c, length_z_c = converter.encode(labels_z_c, batch_max_length=opt.batch_max_length)
        
        z_c_code = cEncoder(text_z_c)
        style = mixing_noise(z_c_code, opt.batch_size, opt.latent, opt.mixing, device)

        if opt.zAlone:
            #to validate orig style gan results
            newstyle = []
            newstyle.append(style[0][:,:opt.latent])
            if len(style)>1:
                newstyle.append(style[1][:,:opt.latent])
            style = newstyle
        
        fake_img,_ = genModel(style, input_is_latent=opt.input_latent)

        fake_pred = disEncModel(fake_img)
        disGenCost = g_nonsaturating_loss(fake_pred)

        if opt.zAlone:
            ocrCost = torch.tensor(0.0)
        else:
            #Compute OCR prediction (Reconstruction of content)
            # text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            # length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(device)
            
            if 'CTC' in opt.Prediction:
                preds_recon = ocrModel(fake_img, text_z_c, is_train=False)
                preds_size = torch.IntTensor([preds_recon.size(1)] * opt.batch_size)
                preds_recon_softmax = preds_recon.log_softmax(2).permute(1, 0, 2)
                ocrCost = ocrCriterion(preds_recon_softmax, text_z_c, preds_size, length_z_c)
            else:
                print("Not implemented error")
                sys.exit()

        gen_enc_cost = disGenCost + opt.ocrWeight * ocrCost
        loss_avg_gen.add(disGenCost)
        loss_avg_ocr_unsup.add(opt.ocrWeight * ocrCost)

        genModel.zero_grad()
        cEncoder.zero_grad()

        gen_enc_cost = disGenCost + opt.ocrWeight * ocrCost
        grad_fake_OCR = torch.autograd.grad(ocrCost, fake_img, retain_graph=True)[0]
        loss_grad_fake_OCR = 10**6*torch.mean(grad_fake_OCR**2)
        grad_fake_adv = torch.autograd.grad(disGenCost, fake_img, retain_graph=True)[0]
        loss_grad_fake_adv = 10**6*torch.mean(grad_fake_adv**2)
        
        if opt.grad_balance:
            gen_enc_cost.backward(retain_graph=True)
            grad_fake_OCR = torch.autograd.grad(ocrCost, fake_img, create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(disGenCost, fake_img, create_graph=True, retain_graph=True)[0]
            a = opt.ocrWeight * torch.div(torch.std(grad_fake_adv), epsilon+torch.std(grad_fake_OCR))
            if a is None:
                print(ocrCost, disGenCost, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
            if a>1000 or a<0.0001:
                print(a)
            
            ocrCost = a.detach() * ocrCost
            gen_enc_cost = disGenCost + ocrCost
            gen_enc_cost.backward(retain_graph=True)
            grad_fake_OCR = torch.autograd.grad(ocrCost, fake_img, create_graph=False, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(disGenCost, fake_img, create_graph=False, retain_graph=True)[0]
            loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
            loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
            with torch.no_grad():
                gen_enc_cost.backward()
        else:
            gen_enc_cost.backward()

        loss_avg_gen.add(disGenCost)
        loss_avg_ocr_unsup.add(opt.ocrWeight * ocrCost)

        optimizer.step()
        
        g_regularize = cntr % opt.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, opt.batch_size // opt.path_batch_shrink)
            # image_input_tensors, _, labels, _ = iter(train_loader).next()
            labels_z_c = iter(text_loader).next()

            # image_input_tensors = image_input_tensors.to(device)
            # gt_image_tensors = image_input_tensors[:path_batch_size]

            # labels_gt = labels[:path_batch_size]

            text_z_c, length_z_c = converter.encode(labels_z_c[:path_batch_size], batch_max_length=opt.batch_max_length)
            # text_gt, length_gt = converter.encode(labels_gt, batch_max_length=opt.batch_max_length)
        
            z_c_code = cEncoder(text_z_c)
            style = mixing_noise(z_c_code, path_batch_size, opt.latent, opt.mixing, device)

            if opt.zAlone:
                #to validate orig style gan results
                newstyle = []
                newstyle.append(style[0][:,:opt.latent])
                if len(style)>1:
                    newstyle.append(style[1][:,:opt.latent])
                style = newstyle

            fake_img, grad = genModel(style, return_latents=True, g_path_regularize=True, mean_path_length=mean_path_length)
            
            decay = 0.01
            path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

            mean_path_length_orig = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
            path_loss = (path_lengths - mean_path_length_orig).pow(2).mean()
            mean_path_length = mean_path_length_orig.detach().item()

            genModel.zero_grad()
            cEncoder.zero_grad()
            weighted_path_loss = opt.path_regularize * opt.g_reg_every * path_loss

            if opt.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            optimizer.step()

            # mean_path_length_avg = (
            #     reduce_sum(mean_path_length).item() / get_world_size()
            # )
            #commented above for multi-gpu , non-distributed setting
            mean_path_length_avg = mean_path_length

        accumulate(g_ema, genModel, accum)

        log_avg_path_loss_val.add(path_loss)
        log_avg_mean_path_length_avg.add(torch.tensor(mean_path_length_avg))
        log_ada_aug_p.add(torch.tensor(ada_aug_p))
        

        if get_rank() == 0:
            if wandb and opt.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,   
                        "Path Length": path_length_val,
                    }
                )
        
        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            
            #generate paired content with similar style
            labels_z_c_1 = iter(text_loader).next()
            labels_z_c_2 = iter(text_loader).next()
            
            text_z_c_1, length_z_c_1 = converter.encode(labels_z_c_1, batch_max_length=opt.batch_max_length)
            text_z_c_2, length_z_c_2 = converter.encode(labels_z_c_2, batch_max_length=opt.batch_max_length)

            z_c_code_1 = cEncoder(text_z_c_1)
            z_c_code_2 = cEncoder(text_z_c_2)

            style_c1_s1 = mixing_noise(z_c_code_1, opt.batch_size, opt.latent, opt.mixing, device)
            style_c2_s1 = []
            style_c2_s1.append(torch.cat((style_c1_s1[0][:,:opt.latent],z_c_code_2),dim=1))
            if len(style_c1_s1)>1:
                style_c2_s1.append(torch.cat((style_c1_s1[1][:,:opt.latent],z_c_code_2),dim=1))
            
            style_c1_s2 = mixing_noise(z_c_code_1, opt.batch_size, opt.latent, opt.mixing, device)
            
            if opt.zAlone:
                #to validate orig style gan results
                newstyle = []
                newstyle.append(style_c1_s1[0][:,:opt.latent])
                if len(style_c1_s1)>1:
                    newstyle.append(style_c1_s1[1][:,:opt.latent])
                style_c1_s1 = newstyle
                style_c2_s1 = newstyle
                style_c1_s2 = newstyle
            
            fake_img_c1_s1, _ = g_ema(style_c1_s1, input_is_latent=opt.input_latent)
            fake_img_c2_s1, _ = g_ema(style_c2_s1, input_is_latent=opt.input_latent)
            fake_img_c1_s2, _ = g_ema(style_c1_s2, input_is_latent=opt.input_latent)

            if not opt.zAlone:
                #Run OCR prediction
                if 'CTC' in opt.Prediction:
                    preds = ocrModel(fake_img_c1_s1, text_z_c_1, is_train=False)
                    preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
                    _, preds_index = preds.max(2)
                    preds_str_fake_img_c1_s1 = converter.decode(preds_index.data, preds_size.data)

                    preds = ocrModel(fake_img_c2_s1, text_z_c_2, is_train=False)
                    preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
                    _, preds_index = preds.max(2)
                    preds_str_fake_img_c2_s1 = converter.decode(preds_index.data, preds_size.data)

                    preds = ocrModel(fake_img_c1_s2, text_z_c_1, is_train=False)
                    preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
                    _, preds_index = preds.max(2)
                    preds_str_fake_img_c1_s2 = converter.decode(preds_index.data, preds_size.data)

                    preds = ocrModel(gt_image_tensors, text_gt, is_train=False)
                    preds_size = torch.IntTensor([preds.size(1)] * gt_image_tensors.shape[0])
                    _, preds_index = preds.max(2)
                    preds_str_gt = converter.decode(preds_index.data, preds_size.data)

                else:
                    print("Not implemented error")
                    sys.exit()
            else:
                preds_str_fake_img_c1_s1 = [':None:'] * fake_img_c1_s1.shape[0]
                preds_str_gt = [':None:'] * fake_img_c1_s1.shape[0] 

            os.makedirs(os.path.join(opt.trainDir,str(iteration)), exist_ok=True)
            for trImgCntr in range(opt.batch_size):
                try:
                    save_image(tensor2im(fake_img_c1_s1[trImgCntr].detach()),os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_c1_s1_'+labels_z_c_1[trImgCntr]+'_ocr:'+preds_str_fake_img_c1_s1[trImgCntr]+'.png'))
                    if not opt.zAlone:
                        save_image(tensor2im(fake_img_c2_s1[trImgCntr].detach()),os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_c2_s1_'+labels_z_c_2[trImgCntr]+'_ocr:'+preds_str_fake_img_c2_s1[trImgCntr]+'.png'))
                        save_image(tensor2im(fake_img_c1_s2[trImgCntr].detach()),os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_c1_s2_'+labels_z_c_1[trImgCntr]+'_ocr:'+preds_str_fake_img_c1_s2[trImgCntr]+'.png'))
                        if trImgCntr<gt_image_tensors.shape[0]:
                            save_image(tensor2im(gt_image_tensors[trImgCntr].detach()),os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_gt_act:'+labels_gt[trImgCntr]+'_ocr:'+preds_str_gt[trImgCntr]+'.png'))
                except:
                    print('Warning while saving training image')
            
            elapsed_time = time.time() - start_time
            # for log
            
            with open(os.path.join(opt.exp_dir,opt.exp_name,'log_train.txt'), 'a') as log:

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}]  \
                    Train Dis loss: {loss_avg_dis.val():0.5f}, Train Gen loss: {loss_avg_gen.val():0.5f},\
                    Train UnSup OCR loss: {loss_avg_ocr_unsup.val():0.5f}, Train Sup OCR loss: {loss_avg_ocr_sup.val():0.5f}, \
                    Train R1-val loss: {log_r1_val.val():0.5f}, Train avg-path-loss: {log_avg_path_loss_val.val():0.5f}, \
                    Train mean-path-length loss: {log_avg_mean_path_length_avg.val():0.5f}, Train ada-aug-p: {log_ada_aug_p.val():0.5f}, \
                    Elapsed_time: {elapsed_time:0.5f}'
                
                
                #plotting
                lib.plot.plot(os.path.join(opt.plotDir,'Train-Dis-Loss'), loss_avg_dis.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-Gen-Loss'), loss_avg_gen.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-UnSup-OCR-Loss'), loss_avg_ocr_unsup.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-Sup-OCR-Loss'), loss_avg_ocr_sup.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-r1_val'), log_r1_val.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-path_loss_val'), log_avg_path_loss_val.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-mean_path_length_avg'), log_avg_mean_path_length_avg.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-ada_aug_p'), log_ada_aug_p.val().item())

                
                print(loss_log)

                loss_avg_dis.reset()
                loss_avg_gen.reset()
                loss_avg_ocr_unsup.reset()
                loss_avg_ocr_sup.reset()
                log_r1_val.reset()
                log_avg_path_loss_val.reset()
                log_avg_mean_path_length_avg.reset()
                log_ada_aug_p.reset()
                

            lib.plot.flush()

        lib.plot.tick()

        # save model per 1e+5 iter.
        if (iteration) % 1e+4 == 0:
            torch.save({
                'cEncoder':cEncoder.state_dict(),
                'genModel':genModel.state_dict(),
                'g_ema':g_ema.state_dict(),
                'ocrModel':ocrModel.state_dict(),
                'disEncModel':disEncModel.state_dict(),
                'optimizer':optimizer.state_dict(),
                'ocr_optimizer':ocr_optimizer.state_dict(),
                'dis_optimizer':dis_optimizer.state_dict()}, 
                os.path.join(opt.exp_dir,opt.exp_name,'iter_'+str(iteration+1)+'_synth.pth'))
            

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
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp13/', help='Where to store logs and models')
    parser.add_argument('--exp_name', default='debug', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--words_file', required=True, default='', help="path to words file. phoc will be sampled from this file")
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=900000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval between each validation')
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
    parser.add_argument('--disWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--gamma_g', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--gamma_e', type=float, default=0.0, help='weights for loss')
    parser.add_argument('--beta', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--alpha', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--vggPerWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--vggStyWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--styleDetach', action='store_true', help='whether to detach style')
    parser.add_argument('--gan_type', default='lsgan', help='lsgan/nsgan/wgan')
    parser.add_argument('--imgReconLoss', default='l1', help='l1/ssim/ms-ssim')
    parser.add_argument('--styleLoss', default='l1', help='l1/triplet')
    parser.add_argument('--contentLoss', default='pred', help='pred(ctc/attn)/vis/seq')
    parser.add_argument('--tripletMargin', type=float, default=1.0, help='triplet margin')
    parser.add_argument('--style_input', action='store_true', help='whether target style input is given for training/validation')
    parser.add_argument('--style_content_input', action='store_true', help='whether target  input content image is given for training/validation')

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
    parser.add_argument("--content_inject_index", type=int, default=1)
    parser.add_argument("--zAlone", action="store_true", help="test original style GAN")
    parser.add_argument("--grad_balance", action="store_true", help="to use gradient balancing")

    opt = parser.parse_args()
    
    if opt.zAlone:
        opt.gamma_e = 0.0
        opt.gamma_g = 0.0
        opt.beta = 0.0

    if opt.rgb:
        opt.input_channel = 3

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name), exist_ok=True)
    opt.trainDir=os.path.join(opt.exp_dir,opt.exp_name,'trainImages')
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
    opt.num_gpu = torch.cuda.device_count()

    train(opt)
