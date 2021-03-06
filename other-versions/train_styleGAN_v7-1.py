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
from dataset import hierarchical_dataset, AlignPairImgCollate, Batch_Balanced_Dataset, tensor2im, save_image, LmdbStyleContentDataset
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


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def train(opt):
    lib.print_model_settings(locals().copy())

    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    log = open(os.path.join(opt.exp_dir,opt.exp_name,'log_dataset.txt'), 'a')
    AlignCollate_valid = AlignPairImgCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    # train_dataset = LmdbStyleDataset(root=opt.train_data, opt=opt)
    
    train_dataset = LmdbStyleContentDataset(root=opt.train_data, opt=opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, #*2 to sample different images from training encoder and discriminator real images
        shuffle=False,  # 'True' to check training progress with validation function.
        sampler=data_sampler(train_dataset, shuffle=True, distributed=opt.distributed),
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True, drop_last=True)
    train_loader = sample_data(train_loader)
    print('-' * 80)
    
    # valid_dataset = LmdbStyleDataset(root=opt.valid_data, opt=opt)
    valid_dataset = LmdbStyleContentDataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, #*2 to sample different images from training encoder and discriminator real images
        shuffle=False,  # 'True' to check training progress with validation function.
        sampler=data_sampler(valid_dataset, shuffle=False, distributed=opt.distributed),
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True, drop_last=True)
    numTestSamples = len(valid_loader)
    valid_loader = sample_data(valid_loader)

    # log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    if 'Attn' in opt.Prediction:
        converter = AttnLabelConverter(opt.character)
    else:
        converter = CTCLabelConverter(opt.character)
    
    opt.num_class = len(converter.character)

    ocrModel = ModelV1(opt).to(device)
    #Currently done to support pre-trained models
    ocrModel = torch.nn.DataParallel(ocrModel).to(device)

    genModel = styleGANGen(opt.size, opt.latent, opt.n_mlp, opt.num_class, channel_multiplier=opt.channel_multiplier).to(device)
    disModel = styleGANDis(opt.size, channel_multiplier=opt.channel_multiplier, input_dim=opt.input_channel).to(device)
    g_ema = styleGANGen(opt.size, opt.latent, opt.n_mlp, opt.num_class, channel_multiplier=opt.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, genModel, 0)

    print('model input parameters', opt.imgH, opt.imgW, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length)

    g_reg_ratio = opt.g_reg_every / (opt.g_reg_every + 1)
    d_reg_ratio = opt.d_reg_every / (opt.d_reg_every + 1)

    optimizer = optim.Adam(
        genModel.parameters(),
        lr=opt.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    dis_optimizer = optim.Adam(
        disModel.parameters(),
        lr=opt.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    ## Loading pre-trained files
    if opt.modelFolderFlag:
        if len(glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth")))>0:
            opt.saved_synth_model = glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth"))[-1]

    if opt.saved_ocr_model !='' and opt.saved_ocr_model !='None':
        print(f'loading pretrained ocr model from {opt.saved_ocr_model}')
        checkpoint = torch.load(opt.saved_ocr_model)
        ocrModel.load_state_dict(checkpoint)

        ocrModel = ocrModel.module

    if opt.saved_synth_model != '' and opt.saved_synth_model != 'None':
        print(f'loading pretrained synth model from {opt.saved_synth_model}')
        checkpoint = torch.load(opt.saved_synth_model)
        genModel.load_state_dict(checkpoint['genModel'])
        g_ema.load_state_dict(checkpoint['g_ema'])
        disModel.load_state_dict(checkpoint['disModel'])
        
        optimizer.load_state_dict(checkpoint["optimizer"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])

    if opt.distributed:
        genModel = torch.nn.parallel.DistributedDataParallel(
            genModel,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )

        disModel = torch.nn.parallel.DistributedDataParallel(
            disModel,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False
        )
        ocrModel = torch.nn.parallel.DistributedDataParallel(
            ocrModel,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )

    if opt.ocrFixed:
        if opt.distributed:
            if opt.Transformation == 'TPS':
                ocrModel.module.Transformation.eval()
            ocrModel.module.FeatureExtraction.eval()
            ocrModel.module.AdaptiveAvgPool.eval()
            # ocrModel.module.SequenceModeling.eval()
            ocrModel.module.Prediction.eval()
        else:
            if opt.Transformation == 'TPS':
                ocrModel.Transformation.eval()
            ocrModel.FeatureExtraction.eval()
            ocrModel.AdaptiveAvgPool.eval()
            # ocrModel.module.SequenceModeling.eval()
            ocrModel.Prediction.eval()
    else:
        ocrModel.train()

    if 'CTC' in opt.Prediction:
        ocrCriterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        ocrCriterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    
    # loss averager
    loss_avg = Averager()
    loss_avg_dis = Averager()
    loss_avg_gen = Averager()
    loss_avg_imgRecon = Averager()
    loss_avg_vgg_per = Averager()
    loss_avg_vgg_sty = Averager()
    loss_avg_ocr = Averager()

    loss_avg_valid = Averager()
    loss_avg_dis_valid = Averager()
    loss_avg_gen_valid = Averager()
    loss_avg_ocr_valid = Averager()

    log_r1_val = Averager()
    log_avg_path_loss_val = Averager()
    log_avg_mean_path_length_avg = Averager()
    log_ada_aug_p = Averager()

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
    loss_dict = {}

    if opt.distributed:
        genModel_module = genModel.module
        disModel_module = disModel.module
        ocrModel_module = ocrModel.module
    else:
        genModel_module = genModel
        disModel_module = disModel
        ocrModel_module = ocrModel


    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = opt.augment_p if opt.augment_p > 0 else 0.0
    ada_aug_step = opt.ada_target / opt.ada_length
    r_t_stat = 0

    sample_z = torch.randn(opt.n_sample, opt.latent, device=device)

    while(True):
        # print(cntr)
        # train part
        if opt.lr_policy !="None":
            scheduler.step()
            dis_scheduler.step()
        
        # image_input_tensors, image_gt_tensors, labels_1, labels_2 = iter(train_loader).next()
        image_input_tensors, image_gt_tensors, labels_1, labels_2,_,_  = next(train_loader)

        image_input_tensors = image_input_tensors.to(device)
        image_gt_tensors = image_gt_tensors.to(device)
        batch_size = image_input_tensors.size(0)

        requires_grad(genModel, False)
        requires_grad(disModel, True)

        text_1, length_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_2, length_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        
        #forward pass from style and word generator
        style = mixing_noise(opt.batch_size, opt.latent, opt.mixing, device)
        
        if 'CTC' in opt.Prediction:
            images_recon_2,_ = genModel(style, text_2, input_is_latent=opt.input_latent)
        else:
            images_recon_2,_ = genModel(style, text_2[:,1:-1], input_is_latent=opt.input_latent)
        
        #Domain discriminator: Dis update
        if opt.augment:
            image_gt_tensors_aug, _ = augment(image_gt_tensors, ada_aug_p)
            images_recon_2, _ = augment(images_recon_2, ada_aug_p)

        else:
            image_gt_tensors_aug = image_gt_tensors

        fake_pred = disModel(images_recon_2)
        real_pred = disModel(image_gt_tensors_aug)
        disCost = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = disCost*opt.disWeight
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        loss_avg_dis.add(disCost)

        disModel.zero_grad()
        disCost.backward()
        dis_optimizer.step()

        if opt.augment and opt.augment_p == 0:
            ada_augment += torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment = reduce_sum(ada_augment)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > opt.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = cntr % opt.d_reg_every == 0

        if d_regularize:
            image_gt_tensors.requires_grad = True
            image_input_tensors.requires_grad = True
            cat_tensor = image_gt_tensors
            real_pred = disModel(cat_tensor)
            
            r1_loss = d_r1_loss(real_pred, cat_tensor)

            disModel.zero_grad()
            (opt.r1 / 2 * r1_loss * opt.d_reg_every + 0 * real_pred[0]).backward()

            dis_optimizer.step()

        loss_dict["r1"] = r1_loss

        # #[Style Encoder] + [Word Generator] update
        # image_input_tensors, image_gt_tensors, labels_1, labels_2 = iter(train_loader).next()
        image_input_tensors, image_gt_tensors, labels_1, labels_2,_,_  = next(train_loader)
        
        image_input_tensors = image_input_tensors.to(device)
        image_gt_tensors = image_gt_tensors.to(device)
        batch_size = image_input_tensors.size(0)

        requires_grad(genModel, True)
        requires_grad(disModel, False)

        text_1, length_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_2, length_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)

        style = mixing_noise(batch_size, opt.latent, opt.mixing, device)

        if 'CTC' in opt.Prediction:
            images_recon_2, _ = genModel(style, text_2)
        else:
            images_recon_2, _ = genModel(style, text_2[:,1:-1])

        if opt.augment:
            images_recon_2, _ = augment(images_recon_2, ada_aug_p)

        fake_pred = disModel(images_recon_2)
        disGenCost = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = disGenCost

        #ocr loss
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)

        if 'CTC' in opt.Prediction:
            preds_recon = ocrModel(images_recon_2, text_for_pred, is_train=False)
            # preds_o = preds_recon[:, :text_1.shape[1], :]
            preds_size = torch.IntTensor([preds_recon.size(1)] * batch_size)
            preds_recon_softmax = preds_recon.log_softmax(2).permute(1, 0, 2)
            ocrCost = ocrCriterion(preds_recon_softmax, text_2, preds_size, length_2)
            
            #predict ocr recognition on generated images
            # preds_recon_size = torch.IntTensor([preds_recon.size(1)] * batch_size)
            _, preds_recon_index = preds_recon.max(2)
            labels_o_ocr = converter.decode(preds_recon_index.data, preds_size.data)

            #predict ocr recognition on gt style images
            preds_s = ocrModel(image_input_tensors, text_for_pred, is_train=False)
            # preds_s = preds_s[:, :text_1.shape[1] - 1, :]
            preds_s_size = torch.IntTensor([preds_s.size(1)] * batch_size)
            _, preds_s_index = preds_s.max(2)
            labels_s_ocr = converter.decode(preds_s_index.data, preds_s_size.data)

            #predict ocr recognition on gt stylecontent images
            preds_sc = ocrModel(image_gt_tensors, text_for_pred, is_train=False)
            # preds_sc = preds_sc[:, :text_2.shape[1] - 1, :]
            preds_sc_size = torch.IntTensor([preds_sc.size(1)] * batch_size)
            _, preds_sc_index = preds_sc.max(2)
            labels_sc_ocr = converter.decode(preds_sc_index.data, preds_sc_size.data)

        else:
            preds_recon = ocrModel(images_recon_2, text_for_pred[:, :-1], is_train=False)  # align with Attention.forward
            target_2 = text_2[:, 1:]  # without [GO] Symbol
            ocrCost = ocrCriterion(preds_recon.view(-1, preds_recon.shape[-1]), target_2.contiguous().view(-1))

            #predict ocr recognition on generated images
            _, preds_o_index = preds_recon.max(2)
            labels_o_ocr = converter.decode(preds_o_index, length_for_pred)
            for idx, pred in enumerate(labels_o_ocr):
                pred_EOS = pred.find('[s]')
                labels_o_ocr[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

            #predict ocr recognition on gt style images
            preds_s = ocrModel(image_input_tensors, text_for_pred, is_train=False)
            _, preds_s_index = preds_s.max(2)
            labels_s_ocr = converter.decode(preds_s_index, length_for_pred)
            for idx, pred in enumerate(labels_s_ocr):
                pred_EOS = pred.find('[s]')
                labels_s_ocr[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            
            #predict ocr recognition on gt stylecontent images
            preds_sc = ocrModel(image_gt_tensors, text_for_pred, is_train=False)
            _, preds_sc_index = preds_sc.max(2)
            labels_sc_ocr = converter.decode(preds_sc_index, length_for_pred)
            for idx, pred in enumerate(labels_sc_ocr):
                pred_EOS = pred.find('[s]')
                labels_sc_ocr[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

        cost =  opt.disWeight*disGenCost + opt.ocrWeight*ocrCost

        genModel.zero_grad()
        disModel.zero_grad()
        ocrModel.zero_grad()
        
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)

        g_regularize = cntr % opt.g_reg_every == 0

        if g_regularize:
            image_input_tensors, image_gt_tensors, labels_1, labels_2, _, _ = next(train_loader)
        
            image_input_tensors = image_input_tensors.to(device)
            image_gt_tensors = image_gt_tensors.to(device)
            batch_size = image_input_tensors.size(0)

            text_1, length_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
            text_2, length_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)

            path_batch_size = max(1, batch_size // opt.path_batch_shrink)

            style = mixing_noise(path_batch_size, opt.latent, opt.mixing, device)
            
            if 'CTC' in opt.Prediction:
                images_recon_2, latents = genModel(style, text_2[:path_batch_size], return_latents=True)
            else:
                images_recon_2, latents = genModel(style, text_2[:path_batch_size,1:-1], return_latents=True)
            
            path_loss, mean_path_length, path_lengths = g_path_regularize(
                images_recon_2, latents, mean_path_length
            )

            genModel.zero_grad()
            weighted_path_loss = opt.path_regularize * opt.g_reg_every * path_loss

            if opt.path_batch_shrink:
                weighted_path_loss += 0 * images_recon_2[0, 0, 0, 0]

            weighted_path_loss.backward()

            optimizer.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        # accumulate(g_ema, genModel, accum)
        accumulate(g_ema, genModel_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()


        #Individual losses
        loss_avg_gen.add(opt.disWeight*disGenCost)
        loss_avg_imgRecon.add(torch.tensor(0.0))
        loss_avg_vgg_per.add(torch.tensor(0.0))
        loss_avg_vgg_sty.add(torch.tensor(0.0))
        loss_avg_ocr.add(opt.ocrWeight*ocrCost)

        log_r1_val.add(loss_reduced["path"])
        log_avg_path_loss_val.add(loss_reduced["path"])
        log_avg_mean_path_length_avg.add(torch.tensor(mean_path_length_avg))
        log_ada_aug_p.add(torch.tensor(ada_aug_p))

        if get_rank() == 0:
            # pbar.set_description(
            #     (
            #         f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
            #         f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
            #         f"augment: {ada_aug_p:.4f}"
            #     )
            # )

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
            
            # if cntr % 100 == 0:
            #     with torch.no_grad():
            #         g_ema.eval()
            #         sample, _ = g_ema([scInput[:,:opt.latent],scInput[:,opt.latent:]])
            #         utils.save_image(
            #             sample,
            #             os.path.join(opt.trainDir, f"sample_{str(cntr).zfill(6)}.png"),
            #             nrow=int(opt.n_sample ** 0.5),
            #             normalize=True,
            #             range=(-1, 1),
            #         )


        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            g_ema.eval()
            disModel.eval()

            #Save validation images
            image_input_tensors, image_gt_tensors, labels_1, labels_2, _, _ = next(valid_loader)
            image_input_tensors = image_input_tensors.to(device)
            image_gt_tensors = image_gt_tensors.to(device)
            batch_size = image_input_tensors.size(0)

            text_1, length_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
            text_2, length_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
            
            style = mixing_noise(opt.batch_size, opt.latent, opt.mixing, device)

            curr_batch_size = style[0].shape[0]

            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)

            if 'CTC' in opt.Prediction:
                images_recon_2, _ = g_ema(style, text_2[:curr_batch_size], input_is_latent=opt.input_latent)

                preds_recon = ocrModel(images_recon_2, text_for_pred, is_train=False)
                preds_recon_softmax = preds_recon.log_softmax(2).permute(1, 0, 2)
                ocrCost = ocrCriterion(preds_recon_softmax, text_2, preds_size, length_2)

                preds_size = torch.IntTensor([preds_recon.size(1)] * batch_size)
                _, preds_recon_index = preds_recon.max(2)
                labels_o_ocr = converter.decode(preds_recon_index.data, preds_size.data)
                

                #predict ocr recognition on gt style images
                preds_s = ocrModel(image_input_tensors, text_for_pred, is_train=False)
                preds_s_size = torch.IntTensor([preds_s.size(1)] * batch_size)
                _, preds_s_index = preds_s.max(2)
                labels_s_ocr = converter.decode(preds_s_index.data, preds_s_size.data)

                #predict ocr recognition on gt stylecontent images
                preds_sc = ocrModel(image_gt_tensors, text_for_pred, is_train=False)
                preds_sc_size = torch.IntTensor([preds_sc.size(1)] * batch_size)
                _, preds_sc_index = preds_sc.max(2)
                labels_sc_ocr = converter.decode(preds_sc_index.data, preds_sc_size.data)

            else:
                images_recon_2, _ = g_ema(style, text_2[:curr_batch_size,1:-1], input_is_latent=opt.input_latent)

                #predict ocr recognition on generated images
                preds_recon = ocrModel(images_recon_2, text_for_pred[:, :-1], is_train=False)  # align with Attention.forward
                target_2 = text_2[:, 1:]  # without [GO] Symbol
                ocrCost = ocrCriterion(preds_recon.view(-1, preds_recon.shape[-1]), target_2.contiguous().view(-1))

                _, preds_o_index = preds_recon.max(2)
                labels_o_ocr = converter.decode(preds_o_index, length_for_pred)
                for idx, pred in enumerate(labels_o_ocr):
                    pred_EOS = pred.find('[s]')
                    labels_o_ocr[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

                #predict ocr recognition on gt style images
                preds_s = ocrModel(image_input_tensors, text_for_pred, is_train=False)
                _, preds_s_index = preds_s.max(2)
                labels_s_ocr = converter.decode(preds_s_index, length_for_pred)
                for idx, pred in enumerate(labels_s_ocr):
                    pred_EOS = pred.find('[s]')
                    labels_s_ocr[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                
                #predict ocr recognition on gt stylecontent images
                preds_sc = ocrModel(image_gt_tensors, text_for_pred, is_train=False)
                _, preds_sc_index = preds_sc.max(2)
                labels_sc_ocr = converter.decode(preds_sc_index, length_for_pred)
                for idx, pred in enumerate(labels_sc_ocr):
                    pred_EOS = pred.find('[s]')
                    labels_sc_ocr[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            
            fake_pred = disModel(images_recon_2)
            real_pred = disModel(image_gt_tensors)
            disCost = d_logistic_loss(real_pred, fake_pred)
            disGenCost = g_nonsaturating_loss(fake_pred)

            loss_avg_dis_valid.add(disCost)
            loss_avg_gen_valid.add(opt.disWeight*disGenCost)
            loss_avg_ocr_valid.add(opt.ocrWeight*ocrCost)
            loss_avg_valid.add(opt.disWeight*disGenCost + opt.ocrWeight*ocrCost)

            disModel.train()

            os.makedirs(os.path.join(opt.valDir, str(iteration)), exist_ok=True)
            for trImgCntr in range(batch_size):
                try:
                    utils.save_image(image_input_tensors[trImgCntr].detach(),os.path.join(opt.valDir,str(iteration),str(trImgCntr)+'_sInput_'+labels_1[trImgCntr]+'_'+labels_s_ocr[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                    utils.save_image(image_gt_tensors[trImgCntr].detach(),os.path.join(opt.valDir,str(iteration),str(trImgCntr)+'_csGT_'+labels_2[trImgCntr]+'_'+labels_sc_ocr[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))
                    utils.save_image(images_recon_2[trImgCntr].detach(),os.path.join(opt.valDir,str(iteration),str(trImgCntr)+'_csRecon_'+labels_2[trImgCntr]+'_'+labels_o_ocr[trImgCntr]+'.png'),nrow=1,normalize=True,range=(-1, 1))

                    # save_image(tensor2im(image_input_tensors[trImgCntr].detach()),os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_sInput_'+labels_1[trImgCntr]+'_'+labels_s_ocr[trImgCntr]+'.png'))
                    # save_image(tensor2im(image_gt_tensors[trImgCntr].detach()),os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_csGT_'+labels_2[trImgCntr]+'_'+labels_sc_ocr[trImgCntr]+'.png'))
                    # save_image(tensor2im(images_recon_2[trImgCntr].detach()),os.path.join(opt.trainDir,str(iteration),str(trImgCntr)+'_csRecon_'+labels_2[trImgCntr]+'_'+labels_o_ocr[trImgCntr]+'.png'))
                except Exception as e:
                    print('Warning while saving training image')
                    print(str(e))
            
            elapsed_time = time.time() - start_time
            # for log
            
            with open(os.path.join(opt.exp_dir,opt.exp_name,'log_train.txt'), 'a') as log:

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train Synth loss: {loss_avg.val():0.5f}, \
                    Train Dis loss: {loss_avg_dis.val():0.5f}, Train Gen loss: {loss_avg_gen.val():0.5f},\
                    Train OCR loss: {loss_avg_ocr.val():0.5f}, \
                    Train R1-val loss: {log_r1_val.val():0.5f}, Train avg-path-loss: {log_avg_path_loss_val.val():0.5f}, \
                    Train mean-path-length loss: {log_avg_mean_path_length_avg.val():0.5f}, Train ada-aug-p: {log_ada_aug_p.val():0.5f}, \
                    Valid Synth loss: {loss_avg_valid.val():0.5f}, \
                    Valid Dis loss: {loss_avg_dis_valid.val():0.5f}, Valid Gen loss: {loss_avg_gen_valid.val():0.5f}, \
                    Valid OCR loss: {loss_avg_ocr_valid.val():0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                
                
                #plotting
                lib.plot.plot(os.path.join(opt.plotDir,'Train-Synth-Loss'), loss_avg.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-Dis-Loss'), loss_avg_dis.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-Gen-Loss'), loss_avg_gen.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-OCR-Loss'), loss_avg_ocr.val().item())

                lib.plot.plot(os.path.join(opt.plotDir,'Train-r1_val'), log_r1_val.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-path_loss_val'), log_avg_path_loss_val.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-mean_path_length_avg'), log_avg_mean_path_length_avg.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Train-ada_aug_p'), log_ada_aug_p.val().item())

                lib.plot.plot(os.path.join(opt.plotDir,'Valid-Synth-Loss'), loss_avg_valid.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Valid-Dis-Loss'), loss_avg_dis_valid.val().item())

                lib.plot.plot(os.path.join(opt.plotDir,'Valid-Gen-Loss'), loss_avg_gen_valid.val().item())
                lib.plot.plot(os.path.join(opt.plotDir,'Valid-OCR-Loss'), loss_avg_ocr_valid.val().item())
                
                print(loss_log)

                loss_avg.reset()
                loss_avg_dis.reset()

                loss_avg_gen.reset()
                loss_avg_imgRecon.reset()
                loss_avg_vgg_per.reset()
                loss_avg_vgg_sty.reset()
                loss_avg_ocr.reset()
                loss_avg_dis_valid.reset()
                loss_avg_gen_valid.reset()
                loss_avg_ocr_valid.reset()
                loss_avg_valid.reset()

                log_r1_val.reset()
                log_avg_path_loss_val.reset()
                log_avg_mean_path_length_avg.reset()
                log_ada_aug_p.reset()
                

            lib.plot.flush()

        lib.plot.tick()

        # save model per 1e+5 iter.
        if (iteration) % 3000 == 0:
            torch.save({
                'genModel':genModel_module.state_dict(),
                'g_ema':g_ema.state_dict(),
                'disModel':disModel_module.state_dict(),
                'optimizer':optimizer.state_dict(),
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
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp15/', help='Where to store logs and models')
    parser.add_argument('--exp_name', default='debug', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
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
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
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

    opt = parser.parse_args()
    
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.distributed = n_gpu > 1

    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

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
    # opt.num_gpu = torch.cuda.device_count()

    train(opt)
