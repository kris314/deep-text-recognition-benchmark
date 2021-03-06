# Desc: orig(v0) + gt of target style image

import os
import sys
import time
import random
import string
import argparse
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import numpy as np
import torchvision.models as models
import pdb

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignPairCollate, Batch_Balanced_Dataset, tensor2im, save_image
from model import ModelV1, StyleTensorEncoder, MsImageDisV2, AdaIN_Tensor_WordGenerator, VGGPerceptualLossModel
from test_synth import validation_synth_v4

import tflib as lib
import tflib.plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
    plotDir = os.path.join(opt.exp_dir,opt.exp_name,'plots')
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
    
    lib.print_model_settings(locals().copy())

    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')


    log = open(os.path.join(opt.exp_dir,opt.exp_name,'log_dataset.txt'), 'a')
    AlignCollate_valid = AlignPairCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    train_dataset, train_dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(train_dataset_log)
    print('-' * 80)

    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=False,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    if 'Attn' in opt.Prediction:
        converter = AttnLabelConverter(opt.character)
    else:
        converter = CTCLabelConverter(opt.character)
    
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    
    ocrModel = ModelV1(opt)
    styleModel = StyleTensorEncoder(input_dim=opt.input_channel)
    genModel = AdaIN_Tensor_WordGenerator(opt)
    disModel = MsImageDisV2(opt)

    if opt.contentLoss == 'vis' or opt.contentLoss == 'seq':
        ocrCriterion = torch.nn.L1Loss()
    else:
        if 'CTC' in opt.Prediction:
            ocrCriterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        else:
            ocrCriterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    vggRecCriterion = torch.nn.L1Loss()
    vggModel = VGGPerceptualLossModel(models.vgg19(pretrained=True), vggRecCriterion)
    
    print('model input parameters', opt.imgH, opt.imgW, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length)

    #  weight initialization
    for currModel in [styleModel, genModel, disModel]:
        for name, param in currModel.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

    styleModel = torch.nn.DataParallel(styleModel).to(device)
    styleModel.train()
    
    genModel = torch.nn.DataParallel(genModel).to(device)
    genModel.train()

    disModel = torch.nn.DataParallel(disModel).to(device)
    disModel.train()

    vggModel = torch.nn.DataParallel(vggModel).to(device)
    vggModel.eval()

    ocrModel = torch.nn.DataParallel(ocrModel).to(device)
    ocrModel.module.Transformation.eval()
    ocrModel.module.FeatureExtraction.eval()
    ocrModel.module.AdaptiveAvgPool.eval()
    # ocrModel.module.SequenceModeling.eval()
    ocrModel.module.Prediction.eval()

    if opt.modelFolderFlag:
        if len(glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth")))>0:
            opt.saved_synth_model = glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_synth.pth"))[-1]

    if opt.saved_ocr_model !='' and opt.saved_ocr_model !='None':
        print(f'loading pretrained ocr model from {opt.saved_ocr_model}')
        checkpoint = torch.load(opt.saved_ocr_model)
        ocrModel.load_state_dict(checkpoint)

    if opt.saved_synth_model != '' and opt.saved_synth_model != 'None':
        print(f'loading pretrained synth model from {opt.saved_synth_model}')
        checkpoint = torch.load(opt.saved_synth_model)
        
        styleModel.load_state_dict(checkpoint['styleModel'])
        genModel.load_state_dict(checkpoint['genModel'])
        disModel.load_state_dict(checkpoint['disModel'])

    if opt.imgReconLoss == 'l1':
        recCriterion = torch.nn.L1Loss()
    elif opt.imgReconLoss == 'ssim':
        recCriterion = ssim
    elif opt.imgReconLoss == 'ms-ssim':
        recCriterion = msssim

    if opt.styleLoss == 'l1':
        styleRecCriterion = torch.nn.L1Loss()
    elif opt.styleLoss == 'triplet':
        styleRecCriterion = torch.nn.TripletMarginLoss(margin=opt.tripletMargin, p=1)
    #for validation; check only positive pairs
    styleTestRecCriterion = torch.nn.L1Loss()
    

    # loss averager
    loss_avg = Averager()
    loss_avg_dis = Averager()
    loss_avg_gen = Averager()
    loss_avg_imgRecon = Averager()
    loss_avg_vgg_per = Averager()
    loss_avg_vgg_sty = Averager()
    loss_avg_ocr = Averager()

    ##---------------------------------------##
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, styleModel.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    for p in filter(lambda p: p.requires_grad, genModel.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable style and generator params num : ', sum(params_num))

    # setup optimizer
    if opt.optim=='adam':
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps, weight_decay=opt.weight_decay)
    print("SynthOptimizer:")
    print(optimizer)

    #filter parameters for Dis training
    dis_filtered_parameters = []
    dis_params_num = []
    for p in filter(lambda p: p.requires_grad, disModel.parameters()):
        dis_filtered_parameters.append(p)
        dis_params_num.append(np.prod(p.size()))
    print('Dis Trainable params num : ', sum(dis_params_num))

    # setup optimizer
    if opt.optim=='adam':
        dis_optimizer = optim.Adam(dis_filtered_parameters, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    else:
        dis_optimizer = optim.Adadelta(dis_filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps, weight_decay=opt.weight_decay)
    print("DisOptimizer:")
    print(dis_optimizer)
    ##---------------------------------------##

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


    while(True):
        # train part
        if opt.lr_policy !="None":
            scheduler.step()
            dis_scheduler.step()

        image_input_tensors, image_gt_tensors, labels_1, labels_2 = iter(train_loader).next()
        
        cntr+=1

        image_input_tensors = image_input_tensors.to(device)
        image_gt_tensors = image_gt_tensors.to(device)
        batch_size = image_input_tensors.size(0)

        text_1, length_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_2, length_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        
        #forward pass from style and word generator
        style = styleModel(image_input_tensors)
        
        images_recon_2 = genModel(style, text_2)
        
        #Domain discriminator: Dis update
        disModel.zero_grad()
        disCost = opt.disWeight*(disModel.module.calc_dis_loss(torch.cat((images_recon_2.detach(),image_input_tensors),dim=1), torch.cat((image_gt_tensors,image_input_tensors),dim=1)))
        
        disCost.backward()
        dis_optimizer.step()
        loss_avg_dis.add(disCost)
        
        # #[Style Encoder] + [Word Generator] update
        #Adversarial loss
        disGenCost = disModel.module.calc_gen_loss(torch.cat((images_recon_2,image_input_tensors),dim=1))

        #Input reconstruction loss
        recCost = recCriterion(images_recon_2,image_gt_tensors)

        #vgg loss
        vggPerCost, vggStyleCost = vggModel(image_gt_tensors, images_recon_2)

        #ocr loss
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        if opt.contentLoss == 'vis' or opt.contentLoss == 'seq':
            preds_recon = ocrModel(images_recon_2, text_for_pred, is_train=False, returnFeat=opt.contentLoss)
            preds_gt = ocrModel(image_gt_tensors, text_for_pred, is_train=False, returnFeat=opt.contentLoss)
            ocrCost = ocrCriterion(preds_recon, preds_gt)
        else:
            if 'CTC' in opt.Prediction:
                preds_recon = ocrModel(images_recon_2, text_for_pred, is_train=False)
                preds_o = preds_recon.deepcopy()[:, :text_1.shape[1] - 1, :]
                preds_size = torch.IntTensor([preds_recon.size(1)] * batch_size)
                preds_recon = preds_recon.log_softmax(2).permute(1, 0, 2)
                ocrCost = ocrCriterion(preds_recon, text_2, preds_size, length_2)
                
                #predict ocr recognition on generated images
                preds_o_size = torch.IntTensor([preds_o.size(1)] * batch_size)
                _, preds_o_index = preds_o.max(2)
                labels_o_ocr = converter.decode(preds_o_index.data, preds_o_size.data)

                #predict ocr recognition on gt style images
                preds_s = ocrModel(image_input_tensors, text_for_pred, is_train=False)
                preds_s = preds_s[:, :text_1.shape[1] - 1, :]
                preds_s_size = torch.IntTensor([preds_s.size(1)] * batch_size)
                _, preds_s_index = preds_s.max(2)
                labels_s_ocr = converter.decode(preds_s_index.data, preds_s_size.data)

                #predict ocr recognition on gt stylecontent images
                preds_sc = ocrModel(image_input_tensors, text_for_pred, is_train=False)
                preds_sc = preds_sc[:, :text_2.shape[1] - 1, :]
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

        cost =  opt.reconWeight*recCost + opt.disWeight*disGenCost + opt.vggPerWeight*vggPerCost + opt.vggStyWeight*vggStyleCost + opt.ocrWeight*ocrCost

        styleModel.zero_grad()
        genModel.zero_grad()
        disModel.zero_grad()
        vggModel.zero_grad()
        ocrModel.zero_grad()
        
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)

        #Individual losses
        loss_avg_gen.add(opt.disWeight*disGenCost)
        loss_avg_imgRecon.add(opt.reconWeight*recCost)
        loss_avg_vgg_per.add(opt.vggPerWeight*vggPerCost)
        loss_avg_vgg_sty.add(opt.vggStyWeight*vggStyleCost)
        loss_avg_ocr.add(opt.ocrWeight*ocrCost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            
            #Save training images
            os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration)), exist_ok=True)
            for trImgCntr in range(batch_size):
                try:
                    if opt.contentLoss == 'vis' or opt.contentLoss == 'seq':
                        save_image(tensor2im(image_input_tensors[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_sInput_'+labels_1[trImgCntr]+'.png'))
                        save_image(tensor2im(image_gt_tensors[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_csGT_'+labels_2[trImgCntr]+'.png'))
                        save_image(tensor2im(images_recon_2[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_csRecon_'+labels_2[trImgCntr]+'.png'))
                    else:
                        save_image(tensor2im(image_input_tensors[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_sInput_'+labels_1[trImgCntr]+'_'+labels_s_ocr[trImgCntr]+'.png'))
                        save_image(tensor2im(image_gt_tensors[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_csGT_'+labels_2[trImgCntr]+'_'+labels_sc_ocr[trImgCntr]+'.png'))
                        save_image(tensor2im(images_recon_2[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_csRecon_'+labels_2[trImgCntr]+'_'+labels_o_ocr[trImgCntr]+'.png'))
                except:
                    print('Warning while saving training image')
            
            elapsed_time = time.time() - start_time
            # for log
            
            with open(os.path.join(opt.exp_dir,opt.exp_name,'log_train.txt'), 'a') as log:
                styleModel.eval()
                genModel.eval()
                disModel.eval()
                
                with torch.no_grad():                    
                    valid_loss, infer_time, length_of_data = validation_synth_v4(
                        iteration, styleModel, genModel, vggModel, ocrModel, disModel, recCriterion, ocrCriterion, valid_loader, converter, opt)
                
                styleModel.train()
                genModel.train()
                disModel.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train Synth loss: {loss_avg.val():0.5f}, \
                    Train Dis loss: {loss_avg_dis.val():0.5f}, Train Gen loss: {loss_avg_gen.val():0.5f},\
                    Train ImgRecon loss: {loss_avg_imgRecon.val():0.5f}, Train VGG-Per loss: {loss_avg_vgg_per.val():0.5f},\
                    Train VGG-Sty loss: {loss_avg_vgg_sty.val():0.5f}, Train OCR loss: {loss_avg_ocr.val():0.5f}, Valid Synth loss: {valid_loss[0]:0.5f}, \
                    Valid Dis loss: {valid_loss[1]:0.5f}, Valid Gen loss: {valid_loss[2]:0.5f}, \
                    Valid ImgRecon loss: {valid_loss[3]:0.5f}, Valid VGG-Per loss: {valid_loss[4]:0.5f}, \
                    Valid VGG-Sty loss: {valid_loss[5]:0.5f}, Valid OCR loss: {valid_loss[6]:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                
                
                #plotting
                lib.plot.plot(os.path.join(plotDir,'Train-Synth-Loss'), loss_avg.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-Dis-Loss'), loss_avg_dis.val().item())
                
                lib.plot.plot(os.path.join(plotDir,'Train-Gen-Loss'), loss_avg_gen.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-ImgRecon1-Loss'), loss_avg_imgRecon.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-VGG-Per-Loss'), loss_avg_vgg_per.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-VGG-Sty-Loss'), loss_avg_vgg_sty.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-OCR-Loss'), loss_avg_ocr.val().item())

                lib.plot.plot(os.path.join(plotDir,'Valid-Synth-Loss'), valid_loss[0].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-Dis-Loss'), valid_loss[1].item())

                lib.plot.plot(os.path.join(plotDir,'Valid-Gen-Loss'), valid_loss[2].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-ImgRecon1-Loss'), valid_loss[3].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-VGG-Per-Loss'), valid_loss[4].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-VGG-Sty-Loss'), valid_loss[5].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-OCR-Loss'), valid_loss[6].item())
                
                print(loss_log)

                loss_avg.reset()
                loss_avg_dis.reset()

                loss_avg_gen.reset()
                loss_avg_imgRecon.reset()
                loss_avg_vgg_per.reset()
                loss_avg_vgg_sty.reset()
                loss_avg_ocr.reset()

            lib.plot.flush()

        lib.plot.tick()

        # save model per 1e+5 iter.
        if (iteration) % 1e+4 == 0:
            torch.save({
                'styleModel':styleModel.state_dict(),
                'genModel':genModel.state_dict(),
                'disModel':disModel.state_dict()}, 
                os.path.join(opt.exp_dir,opt.exp_name,'iter_'+str(iteration+1)+'_synth.pth'))
            

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


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
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp10/', help='Where to store logs and models')
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
    parser.add_argument('--saved_dis_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--optim', default='adadelta', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
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
    parser.add_argument('--batch_exact_length', type=int, default=5, help='exact-label-length')
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

    parser.add_argument('--debugFlag', action='store_true', help='for debugging')
    parser.add_argument('--modelFolderFlag', action='store_true', help='load latest files from saved model folder')
    parser.add_argument('--testFlag', action='store_true', help='for testing')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name), exist_ok=True)
    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'trainImages'), exist_ok=True)
    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'valImages'), exist_ok=True)


    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
