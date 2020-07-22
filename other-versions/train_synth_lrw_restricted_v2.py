import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import numpy as np

import pdb

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset, tensor2im, save_image
from model import Model, AdaINGenV2, MsImageDis
from test_synth import validation, validation_synth, validation_synth_adv, validation_synth_lrw, validation_synth_lrw_res

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

    #considering the real images for discriminator
    opt.batch_size = opt.batch_size*2

    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(os.path.join(opt.exp_dir,opt.exp_name,'log_dataset.txt'), 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
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
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    
    model = AdaINGenV2(opt)
    ocrModel = Model(opt)
    disModel = MsImageDis(opt)
    
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    
    #  weight initialization
    for currModel in [model, ocrModel, disModel]:
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
    
    
    # data parallel for multi-GPU
    ocrModel = torch.nn.DataParallel(ocrModel).to(device)
    if not opt.ocrFixed:
        ocrModel.train()
    else:
        ocrModel.module.Transformation.eval()
        ocrModel.module.FeatureExtraction.eval()
        ocrModel.module.AdaptiveAvgPool.eval()
        # ocrModel.module.SequenceModeling.eval()
        ocrModel.module.Prediction.eval()

    model = torch.nn.DataParallel(model).to(device)
    model.train()
    
    disModel = torch.nn.DataParallel(disModel).to(device)
    disModel.train()

    #loading pre-trained model
    if opt.saved_ocr_model != '' and opt.saved_ocr_model != 'None':
        print(f'loading pretrained ocr model from {opt.saved_ocr_model}')
        if opt.FT:
            ocrModel.load_state_dict(torch.load(opt.saved_ocr_model), strict=False)
        else:
            ocrModel.load_state_dict(torch.load(opt.saved_ocr_model))
    print("OCRModel:")
    print(ocrModel)

    if opt.saved_synth_model != '' and opt.saved_synth_model != 'None':
        print(f'loading pretrained synth model from {opt.saved_synth_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_synth_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_synth_model))
    print("SynthModel:")
    print(model)

    if opt.saved_dis_model != '' and opt.saved_dis_model != 'None':
        print(f'loading pretrained discriminator model from {opt.saved_dis_model}')
        if opt.FT:
            disModel.load_state_dict(torch.load(opt.saved_dis_model), strict=False)
        else:
            disModel.load_state_dict(torch.load(opt.saved_dis_model))
    print("DisModel:")
    print(disModel)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        ocrCriterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        ocrCriterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    
    recCriterion = torch.nn.L1Loss()
    styleRecCriterion = torch.nn.L1Loss()

    # loss averager
    loss_avg_ocr = Averager()
    loss_avg = Averager()
    loss_avg_dis = Averager()

    loss_avg_ocrRecon_1 = Averager()
    loss_avg_ocrRecon_2 = Averager()
    loss_avg_gen = Averager()
    loss_avg_imgRecon = Averager()
    loss_avg_styRecon = Averager()

    ##---------------------------------------##
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.optim=='adam':
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps, weight_decay=opt.weight_decay)
    print("SynthOptimizer:")
    print(optimizer)


    # filter that only require gradient decent
    gen_filtered_parameters = []
    gen_params_num = []
    # for p in filter(lambda p: p.requires_grad, model.parameters()):
    for name, p in model.named_parameters():
        if p.requires_grad and not('enc_style' in name):
            gen_filtered_parameters.append(p)
            gen_params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(gen_params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.optim=='adam':
        gen_optimizer = optim.Adam(gen_filtered_parameters, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    else:
        gen_optimizer = optim.Adadelta(gen_filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps, weight_decay=opt.weight_decay)
    print("GenOptimizer:")
    print(gen_optimizer)
    

    #filter parameters for OCR training
    ocr_filtered_parameters = []
    ocr_params_num = []
    for p in filter(lambda p: p.requires_grad, ocrModel.parameters()):
        ocr_filtered_parameters.append(p)
        ocr_params_num.append(np.prod(p.size()))
    print('OCR Trainable params num : ', sum(ocr_params_num))


    # setup optimizer
    if opt.optim=='adam':
        ocr_optimizer = optim.Adam(ocr_filtered_parameters, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    else:
        ocr_optimizer = optim.Adadelta(ocr_filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps, weight_decay=opt.weight_decay)
    print("OCROptimizer:")
    print(ocr_optimizer)

    #filter parameters for OCR training
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
    if opt.saved_synth_model != '':
        try:
            start_iter = int(opt.saved_synth_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    
    #get schedulers
    scheduler = get_scheduler(optimizer,opt)
    ocr_scheduler = get_scheduler(ocr_optimizer,opt)
    dis_scheduler = get_scheduler(dis_optimizer,opt)
    gen_scheduler = get_scheduler(gen_optimizer,opt)

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    best_accuracy_ocr = -1
    best_norm_ED_ocr = -1
    iteration = start_iter
    cntr=0
    while(True):
        # train part
        # pdb.set_trace()
        if opt.lr_policy !="None":
            scheduler.step()
            ocr_scheduler.step()
            dis_scheduler.step()
            gen_scheduler.step()
            
        image_tensors_all, labels_1_all, labels_2_all = train_dataset.get_batch()
        
        # ## comment
        # pdb.set_trace()
        # for imgCntr in range(image_tensors.shape[0]):
        #     save_image(tensor2im(image_tensors[imgCntr]),'temp/'+str(imgCntr)+'.png')
        # pdb.set_trace()
        # ###
        # print(cntr)
        cntr+=1
        disCnt = int(image_tensors_all.size(0)/2)
        image_tensors, image_tensors_real, labels_gt, labels_2 = image_tensors_all[:disCnt], image_tensors_all[disCnt:disCnt+disCnt], labels_1_all[:disCnt], labels_2_all[:disCnt]

        image = image_tensors.to(device)
        image_real = image_tensors_real.to(device)
        batch_size = image.size(0)

        
        ##-----------------------------------##
        #generate text(labels) from ocr.forward
        if opt.ocrFixed:
            # ocrModel.eval()
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            
            if 'CTC' in opt.Prediction:
                preds = ocrModel(image, text_for_pred)
                preds = preds[:, :text_for_loss.shape[1] - 1, :]
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                labels_1 = converter.decode(preds_index.data, preds_size.data)
            else:
                preds = ocrModel(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                labels_1 = converter.decode(preds_index, length_for_pred)
                for idx, pred in enumerate(labels_1):
                    pred_EOS = pred.find('[s]')
                    labels_1[idx] = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            # ocrModel.train()
        else:
            labels_1 = labels_gt
        
        ##-----------------------------------##

        text_1, length_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_2, length_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        
        #forward pass from style and word generator
        images_recon_1, images_recon_2, style = model(image, text_1, text_2)

        if 'CTC' in opt.Prediction:
            
            if not opt.ocrFixed:
                #ocr training with orig image
                preds_ocr = ocrModel(image, text_1)
                preds_size_ocr = torch.IntTensor([preds_ocr.size(1)] * batch_size)
                preds_ocr = preds_ocr.log_softmax(2).permute(1, 0, 2)

                ocrCost_train = ocrCriterion(preds_ocr, text_1, preds_size_ocr, length_1)

            
            #content loss for reconstructed images
            preds_1 = ocrModel(images_recon_1, text_1)
            preds_size_1 = torch.IntTensor([preds_1.size(1)] * batch_size)
            preds_1 = preds_1.log_softmax(2).permute(1, 0, 2)

            preds_2 = ocrModel(images_recon_2, text_2)
            preds_size_2 = torch.IntTensor([preds_2.size(1)] * batch_size)
            preds_2 = preds_2.log_softmax(2).permute(1, 0, 2)
            ocrCost_1 = ocrCriterion(preds_1, text_1, preds_size_1, length_1)
            ocrCost_2 = ocrCriterion(preds_2, text_2, preds_size_2, length_2)
            # ocrCost = 0.5*( ocrCost_1 + ocrCost_2 )

        else:
            if not opt.ocrFixed:
                #ocr training with orig image
                preds_ocr = ocrModel(image, text_1[:, :-1])  # align with Attention.forward
                target_ocr = text_1[:, 1:]  # without [GO] Symbol

                ocrCost_train = ocrCriterion(preds_ocr.view(-1, preds_ocr.shape[-1]), target_ocr.contiguous().view(-1))

            #content loss for reconstructed images
            preds_1 = ocrModel(images_recon_1, text_1[:, :-1], is_train=False)  # align with Attention.forward
            target_1 = text_1[:, 1:]  # without [GO] Symbol

            preds_2 = ocrModel(images_recon_2, text_2[:, :-1], is_train=False)  # align with Attention.forward
            target_2 = text_2[:, 1:]  # without [GO] Symbol

            ocrCost_1 = ocrCriterion(preds_1.view(-1, preds_1.shape[-1]), target_1.contiguous().view(-1))
            ocrCost_2 = ocrCriterion(preds_2.view(-1, preds_2.shape[-1]), target_2.contiguous().view(-1))
            # ocrCost = 0.5*(ocrCost_1+ocrCost_2)
        
        if not opt.ocrFixed:
            #training OCR
            ocrModel.zero_grad()
            ocrCost_train.backward()
            # torch.nn.utils.clip_grad_norm_(ocrModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            ocr_optimizer.step()
            #if ocr is fixed; ignore this loss
            loss_avg_ocr.add(ocrCost_train)
        else:
            loss_avg_ocr.add(torch.tensor(0.0))

        
        #Domain discriminator: Dis update
        disCost = opt.disWeight*0.5*(disModel.module.calc_dis_loss(images_recon_1.detach(), image_real) + disModel.module.calc_dis_loss(images_recon_2.detach(), image))
        disModel.zero_grad()
        disCost.backward()
        # torch.nn.utils.clip_grad_norm_(disModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        dis_optimizer.step()
        loss_avg_dis.add(disCost)
        
        # #[Style Encoder] + [Word Generator] update
        #Adversarial loss
        disGenCost = 0.5*(disModel.module.calc_gen_loss(images_recon_1)+disModel.module.calc_gen_loss(images_recon_2))

        #Input reconstruction loss
        recCost = recCriterion(images_recon_1,image)

        #OCR Content cost
        ocrCost = 0.5*(ocrCost_1+ocrCost_2)

        cost = opt.ocrWeight*ocrCost + opt.reconWeight*recCost + opt.disWeight*disGenCost

        model.zero_grad()
        ocrModel.zero_grad()
        disModel.zero_grad()
        cost.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        #Pair style reconstruction loss
        if opt.styleReconWeight == 0.0:
            styleRecCost = torch.tensor(0.0)
        else:
            # if opt.styleDetach:
            #     styleRecCost = styleRecCriterion(model(images_recon_2, None, None, styleFlag=True), style.detach())
            # else:
            #     styleRecCost = styleRecCriterion(model(images_recon_2, None, None, styleFlag=True), style)
            # with torch.no_grad():
            predStyle = model(images_recon_2, None, None, styleFlag=True)
            styleRecCost =  styleRecCriterion(predStyle, style.detach()) 
        
        model.zero_grad()
        styleRecCost.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        gen_optimizer.step()

        loss_avg.add(cost)

        #Individual losses
        loss_avg_ocrRecon_1.add(opt.ocrWeight*0.5*ocrCost_1)
        loss_avg_ocrRecon_2.add(opt.ocrWeight*0.5*ocrCost_2)
        loss_avg_gen.add(opt.disWeight*disGenCost)
        loss_avg_imgRecon.add(opt.reconWeight*recCost)
        loss_avg_styRecon.add(styleRecCost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            
            #Save training images
            os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration)), exist_ok=True)
            for trImgCntr in range(batch_size):
                try:
                    save_image(tensor2im(image[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_input_'+labels_gt[trImgCntr]+'.png'))
                    save_image(tensor2im(images_recon_1[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_recon_'+labels_1[trImgCntr]+'.png'))
                    save_image(tensor2im(images_recon_2[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_pair_'+labels_2[trImgCntr]+'.png'))
                except:
                    print('Warning while saving training image')
            
            elapsed_time = time.time() - start_time
            # for log
            
            with open(os.path.join(opt.exp_dir,opt.exp_name,'log_train.txt'), 'a') as log:
                model.eval()
                ocrModel.module.Transformation.eval()
                ocrModel.module.FeatureExtraction.eval()
                ocrModel.module.AdaptiveAvgPool.eval()
                ocrModel.module.SequenceModeling.eval()
                ocrModel.module.Prediction.eval()
                disModel.eval()
                
                with torch.no_grad():                    
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation_synth_lrw_res(
                        iteration, model, ocrModel, disModel, recCriterion, styleRecCriterion, ocrCriterion, valid_loader, converter, opt)
                model.train()
                if not opt.ocrFixed:
                    ocrModel.train()
                else:
                #     ocrModel.module.Transformation.eval()
                #     ocrModel.module.FeatureExtraction.eval()
                #     ocrModel.module.AdaptiveAvgPool.eval()
                    ocrModel.module.SequenceModeling.train()
                #     ocrModel.module.Prediction.eval()

                disModel.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train OCR loss: {loss_avg_ocr.val():0.5f}, Train Synth loss: {loss_avg.val():0.5f}, Train Dis loss: {loss_avg_dis.val():0.5f}, Valid OCR loss: {valid_loss[0]:0.5f}, Valid Synth loss: {valid_loss[1]:0.5f}, Valid Dis loss: {valid_loss[2]:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                

                current_model_log_ocr = f'{"Current_accuracy_OCR":17s}: {current_accuracy[0]:0.3f}, {"Current_norm_ED_OCR":17s}: {current_norm_ED[0]:0.2f}'
                current_model_log_1 = f'{"Current_accuracy_recon":17s}: {current_accuracy[1]:0.3f}, {"Current_norm_ED_recon":17s}: {current_norm_ED[1]:0.2f}'
                current_model_log_2 = f'{"Current_accuracy_pair":17s}: {current_accuracy[2]:0.3f}, {"Current_norm_ED_pair":17s}: {current_norm_ED[2]:0.2f}'
                
                #plotting
                lib.plot.plot(os.path.join(plotDir,'Train-OCR-Loss'), loss_avg_ocr.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-Synth-Loss'), loss_avg.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-Dis-Loss'), loss_avg_dis.val().item())
                
                lib.plot.plot(os.path.join(plotDir,'Train-OCR-Recon1-Loss'), loss_avg_ocrRecon_1.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-OCR-Recon2-Loss'), loss_avg_ocrRecon_2.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-Gen-Loss'), loss_avg_gen.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-ImgRecon1-Loss'), loss_avg_imgRecon.val().item())
                lib.plot.plot(os.path.join(plotDir,'Train-StyRecon2-Loss'), loss_avg_styRecon.val().item())

                lib.plot.plot(os.path.join(plotDir,'Valid-OCR-Loss'), valid_loss[0].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-Synth-Loss'), valid_loss[1].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-Dis-Loss'), valid_loss[2].item())

                lib.plot.plot(os.path.join(plotDir,'Valid-OCR-Recon1-Loss'), valid_loss[3].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-OCR-Recon2-Loss'), valid_loss[4].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-Gen-Loss'), valid_loss[5].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-ImgRecon1-Loss'), valid_loss[6].item())
                lib.plot.plot(os.path.join(plotDir,'Valid-StyRecon2-Loss'), valid_loss[7].item())

                lib.plot.plot(os.path.join(plotDir,'Orig-OCR-WordAccuracy'), current_accuracy[0])
                lib.plot.plot(os.path.join(plotDir,'Recon-OCR-WordAccuracy'), current_accuracy[1])
                lib.plot.plot(os.path.join(plotDir,'Pair-OCR-WordAccuracy'), current_accuracy[2])

                lib.plot.plot(os.path.join(plotDir,'Orig-OCR-CharAccuracy'), current_norm_ED[0])
                lib.plot.plot(os.path.join(plotDir,'Recon-OCR-CharAccuracy'), current_norm_ED[1])
                lib.plot.plot(os.path.join(plotDir,'Pair-OCR-CharAccuracy'), current_norm_ED[2])
                

                # keep best accuracy model (on valid dataset)
                if current_accuracy[1] > best_accuracy:
                    best_accuracy = current_accuracy[1]
                    torch.save(model.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'best_accuracy.pth'))
                    torch.save(disModel.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'best_accuracy_dis.pth'))
                if current_norm_ED[1] > best_norm_ED:
                    best_norm_ED = current_norm_ED[1]
                    torch.save(model.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'best_norm_ED.pth'))
                    torch.save(disModel.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'best_norm_ED_dis.pth'))
                best_model_log = f'{"Best_accuracy_Recon":17s}: {best_accuracy:0.3f}, {"Best_norm_ED_Recon":17s}: {best_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy[0] > best_accuracy_ocr:
                    best_accuracy_ocr = current_accuracy[0]
                    if not opt.ocrFixed:
                        torch.save(ocrModel.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'best_accuracy_ocr.pth'))
                if current_norm_ED[0] > best_norm_ED_ocr:
                    best_norm_ED_ocr = current_norm_ED[0]
                    if not opt.ocrFixed:
                        torch.save(ocrModel.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'best_norm_ED_ocr.pth'))
                best_model_log_ocr = f'{"Best_accuracy_ocr":17s}: {best_accuracy_ocr:0.3f}, {"Best_norm_ED_ocr":17s}: {best_norm_ED_ocr:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log_ocr}\n{current_model_log_1}\n{current_model_log_2}\n{best_model_log_ocr}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":32s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                for gt_ocr, pred_ocr, confidence_ocr, gt_1, pred_1, confidence_1, gt_2, pred_2, confidence_2 in zip(labels[0][:5], preds[0][:5], confidence_score[0][:5], labels[1][:5], preds[1][:5], confidence_score[1][:5], labels[2][:5], preds[2][:5], confidence_score[2][:5]):
                    if 'Attn' in opt.Prediction:
                        # gt_ocr = gt_ocr[:gt_ocr.find('[s]')]
                        pred_ocr = pred_ocr[:pred_ocr.find('[s]')]

                        # gt_1 = gt_1[:gt_1.find('[s]')]
                        pred_1 = pred_1[:pred_1.find('[s]')]

                        # gt_2 = gt_2[:gt_2.find('[s]')]
                        pred_2 = pred_2[:pred_2.find('[s]')]

                    predicted_result_log += f'{"ocr"}: {gt_ocr:27s} | {pred_ocr:25s} | {confidence_ocr:0.4f}\t{str(pred_ocr == gt_ocr)}\n'
                    predicted_result_log += f'{"recon"}: {gt_1:25s} | {pred_1:25s} | {confidence_1:0.4f}\t{str(pred_1 == gt_1)}\n'
                    predicted_result_log += f'{"pair"}: {gt_2:26s} | {pred_2:25s} | {confidence_2:0.4f}\t{str(pred_2 == gt_2)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

                loss_avg_ocr.reset()
                loss_avg.reset()
                loss_avg_dis.reset()

                loss_avg_ocrRecon_1.reset()
                loss_avg_ocrRecon_2.reset()
                loss_avg_gen.reset()
                loss_avg_imgRecon.reset()
                loss_avg_styRecon.reset()

            lib.plot.flush()

        lib.plot.tick()

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'iter_'+str(iteration+1)+'.pth'))
            if not opt.ocrFixed:
                torch.save(
                    ocrModel.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'iter_'+str(iteration+1)+'_ocr.pth'))
            torch.save(
                disModel.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'iter_'+str(iteration+1)+'_dis.pth'))

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
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp06/', help='Where to store logs and models')
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
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
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
    parser.add_argument('--styleReconWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--styleDetach', action='store_true', help='whether to detach style')


    parser.add_argument('--debugFlag', action='store_true', help='for debugging')
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
