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
import numpy as np

import pdb

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset, tensor2im, save_image
from model import Model, AdaINGen, MsImageDis
from test_synth import validation, validation_synth, validation_synth_adv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
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
        shuffle=True,  # 'True' to check training progress with validation function.
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
    
    model = AdaINGen(opt)
    ocrModel = Model(opt)
    disModel = MsImageDis()
    
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # Synthesizer weight initialization
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
            if 'weight' in name:
                param.data.fill_(1)
            continue
    
    # Recognizer weight initialization
    for name, param in ocrModel.named_parameters():
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
    
    # Discriminator weight initialization
    for name, param in disModel.named_parameters():
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
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    ocrModel = torch.nn.DataParallel(ocrModel).to(device)
    ocrModel.train()
    
    disModel = torch.nn.DataParallel(disModel).to(device)
    disModel.train()

    
    if opt.saved_synth_model != '':
        print(f'loading pretrained synth model from {opt.saved_synth_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_synth_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_synth_model))
    print("Model:")
    print(model)

    if opt.saved_ocr_model != '':
        print(f'loading pretrained ocr model from {opt.saved_ocr_model}')
        if opt.FT:
            ocrModel.load_state_dict(torch.load(opt.saved_ocr_model), strict=False)
        else:
            ocrModel.load_state_dict(torch.load(opt.saved_ocr_model))
    # ocrModel.eval()   #as we can't call RNN.backward in eval mode
    print("OCRModel:")
    print(ocrModel)

    if opt.saved_dis_model != '':
        print(f'loading pretrained discriminator model from {opt.saved_dis_model}')
        if opt.FT:
            disModel.load_state_dict(torch.load(opt.saved_dis_model), strict=False)
        else:
            disModel.load_state_dict(torch.load(opt.saved_dis_model))
    # ocrModel.eval()   #as we can't call RNN.backward in eval mode
    print("DisModel:")
    print(disModel)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        ocrCriterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        ocrCriterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    
    recCriterion = torch.nn.L1Loss()

    # loss averager
    loss_avg = Averager()
    loss_avg_ocr = Averager() ##----------
    loss_avg_dis = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("SynthOptimizer:")
    print(optimizer)

    #filter parameters for OCR training
    # filter that only require gradient decent
    ocr_filtered_parameters = []
    ocr_params_num = []
    for p in filter(lambda p: p.requires_grad, ocrModel.parameters()):
        ocr_filtered_parameters.append(p)
        ocr_params_num.append(np.prod(p.size()))
    print('OCR Trainable params num : ', sum(ocr_params_num))


    # setup optimizer
    if opt.adam:
        ocr_optimizer = optim.Adam(ocr_filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        ocr_optimizer = optim.Adadelta(ocr_filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("OCROptimizer:")
    print(ocr_optimizer)

    #filter parameters for OCR training
    # filter that only require gradient decent
    dis_filtered_parameters = []
    dis_params_num = []
    for p in filter(lambda p: p.requires_grad, disModel.parameters()):
        dis_filtered_parameters.append(p)
        dis_params_num.append(np.prod(p.size()))
    print('Dis Trainable params num : ', sum(dis_params_num))

    # setup optimizer
    if opt.adam:
        dis_optimizer = optim.Adam(dis_filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        dis_optimizer = optim.Adadelta(dis_filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("DisOptimizer:")
    print(dis_optimizer)



    """ final options """
    # print(opt)
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

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    best_accuracy_ocr = -1
    best_norm_ED_ocr = -1
    iteration = start_iter

    while(True):
        # train part
        
        image_tensors_all, labels_1_all, labels_2_all = train_dataset.get_batch()

        # ## comment
        # pdb.set_trace()
        # for imgCntr in range(image_tensors.shape[0]):
        #     save_image(tensor2im(image_tensors[imgCntr]),'temp/'+str(imgCntr)+'.png')
        # pdb.set_trace()
        # ###
        
        disCnt = int(image_tensors_all.size(0)/2)
        image_tensors, image_tensors_real, labels_1, labels_2 = image_tensors_all[:disCnt], image_tensors_all[disCnt:disCnt+disCnt], labels_1_all[:disCnt], labels_2_all[:disCnt]

        image = image_tensors.to(device)
        image_real = image_tensors_real.to(device)
        text_1, length_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_2, length_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        images_recon_1, images_recon_2, _ = model(image, text_1, text_2)
        

        if 'CTC' in opt.Prediction:
            
            #ocr training
            preds_ocr = ocrModel(image, text_1)
            preds_size_ocr = torch.IntTensor([preds_ocr.size(1)] * batch_size)
            preds_ocr = preds_ocr.log_softmax(2).permute(1, 0, 2)

            ocrCost_train = ocrCriterion(preds_ocr, text_1, preds_size_ocr, length_1)

            #dis training
            #Check: Using alternate real images
            disCost = opt.disWeight*0.5*(disModel.module.calc_dis_loss(images_recon_1.detach(), image_real) + disModel.module.calc_dis_loss(images_recon_2.detach(), image))

            #synth training
            preds_1 = ocrModel(images_recon_1, text_1)
            preds_size_1 = torch.IntTensor([preds_1.size(1)] * batch_size)
            preds_1 = preds_1.log_softmax(2).permute(1, 0, 2)

            preds_2 = ocrModel(images_recon_2, text_2)
            preds_size_2 = torch.IntTensor([preds_2.size(1)] * batch_size)
            preds_2 = preds_2.log_softmax(2).permute(1, 0, 2)

            ocrCost = 0.5*(ocrCriterion(preds_1, text_1, preds_size_1, length_1) + ocrCriterion(preds_2, text_2, preds_size_2, length_2))
            
            #gen training
            disGenCost = 0.5*(disModel.module.calc_gen_loss(images_recon_1)+disModel.module.calc_gen_loss(images_recon_2))

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            ocrCost = ocrCriterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        
        recCost = recCriterion(images_recon_1,image)

        cost = opt.ocrWeight*ocrCost + opt.reconWeight*recCost + opt.disWeight*disGenCost

        disModel.zero_grad()
        disCost.backward()
        torch.nn.utils.clip_grad_norm_(disModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        dis_optimizer.step()

        loss_avg_dis.add(disCost)
        
        model.zero_grad()
        ocrModel.zero_grad()
        disModel.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        #training OCR
        ocrModel.zero_grad()
        ocrCost_train.backward()
        torch.nn.utils.clip_grad_norm_(ocrModel.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        ocr_optimizer.step()

        loss_avg_ocr.add(ocrCost_train)

        #START HERE
        # validation part
        
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            
            #Save training images
            os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration)), exist_ok=True)
            for trImgCntr in range(batch_size):
                try:
                    save_image(tensor2im(image[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_input_'+labels_1[trImgCntr]+'.png'))
                    save_image(tensor2im(images_recon_1[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_recon_'+labels_1[trImgCntr]+'.png'))
                    save_image(tensor2im(images_recon_2[trImgCntr].detach()),os.path.join(opt.exp_dir,opt.exp_name,'trainImages',str(iteration),str(trImgCntr)+'_pair_'+labels_2[trImgCntr]+'.png'))
                except:
                    print('Warning while saving training image')
            
            elapsed_time = time.time() - start_time
            # for log
            
            with open(os.path.join(opt.exp_dir,opt.exp_name,'log_train.txt'), 'a') as log:
                model.eval()
                ocrModel.eval()
                disModel.eval()
                with torch.no_grad():
                    # valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                    #     model, criterion, valid_loader, converter, opt)
                    
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation_synth_adv(
                        iteration, model, ocrModel, disModel, recCriterion, ocrCriterion, valid_loader, converter, opt)
                model.train()
                ocrModel.train()
                disModel.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train OCR loss: {loss_avg_ocr.val():0.5f}, Train Synth loss: {loss_avg.val():0.5f}, Train Dis loss: {loss_avg_dis.val():0.5f}, Valid OCR loss: {valid_loss[0]:0.5f}, Valid Synth loss: {valid_loss[1]:0.5f}, Valid Dis loss: {valid_loss[2]:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg_ocr.reset()
                loss_avg.reset()
                loss_avg_dis.reset()

                current_model_log_ocr = f'{"Current_accuracy_OCR":17s}: {current_accuracy[0]:0.3f}, {"Current_norm_ED_OCR":17s}: {current_norm_ED[0]:0.2f}'
                current_model_log_1 = f'{"Current_accuracy_recon":17s}: {current_accuracy[1]:0.3f}, {"Current_norm_ED_recon":17s}: {current_norm_ED[1]:0.2f}'
                current_model_log_2 = f'{"Current_accuracy_pair":17s}: {current_accuracy[2]:0.3f}, {"Current_norm_ED_pair":17s}: {current_norm_ED[2]:0.2f}'
                
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
                    torch.save(ocrModel.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'best_accuracy_ocr.pth'))
                if current_norm_ED[0] > best_norm_ED_ocr:
                    best_norm_ED_ocr = current_norm_ED[0]
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
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{"ocr"}: {gt_ocr:27s} | {pred_ocr:25s} | {confidence_ocr:0.4f}\t{str(pred_ocr == gt_ocr)}\n'
                    predicted_result_log += f'{"recon"}: {gt_1:25s} | {pred_1:25s} | {confidence_1:0.4f}\t{str(pred_1 == gt_1)}\n'
                    predicted_result_log += f'{"pair"}: {gt_2:26s} | {pred_2:25s} | {confidence_2:0.4f}\t{str(pred_2 == gt_2)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'iter_{iteration+1}.pth'))
            torch.save(
                ocrModel.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'iter_{iteration+1}_ocr.pth'))
            torch.save(
                disModel.state_dict(), os.path.join(opt.exp_dir,opt.exp_name,'iter_{iteration+1}_dis.pth'))

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


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
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=24, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=48, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=288, help='the width of the input image')
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
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--char_embed_size', type=int, default=60, help='character embedding for content encoder')
    parser.add_argument('--ocrWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--reconWeight', type=float, default=1.0, help='weights for loss')
    parser.add_argument('--disWeight', type=float, default=1.0, help='weights for loss')


    parser.add_argument('--debugFlag', action='store_true', help='for debugging')

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
