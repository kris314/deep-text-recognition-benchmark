import os
import time
import string
import argparse
import re
import random

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, tensor2im, save_image
from model import Model, AdaINGenV4, MsImageDisV1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pdb

def benchmark_all_eval(synthModel, ocrModel, recCriterion, styleRecCriterion, ocrCriterion, converter, opt, calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    if calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        log.write(eval_data_log)
        print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
        log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
        print(dashed_line)
        log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()

    return None


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        
        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data

def validation_synth(iterCntr, synthModel, ocrModel, recCriterion, ocrCriterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr)), exist_ok=True)

    n_correct_ocr = 0
    norm_ED_ocr = 0

    n_correct_1 = 0
    norm_ED_1 = 0
    n_correct_2 = 0
    norm_ED_2 = 0

    length_of_data = 0
    infer_time = 0
    valid_loss_avg_ocr = Averager()
    valid_loss_avg = Averager()

    lexicons=[]
    out_of_char = f'[^{opt.character}]'
    #read lexicons file
    with open(opt.lexFile,'r') as lexF:
        for line in lexF:
            lexWord = line[:-1]
            if len(lexWord) <= opt.batch_max_length and not(re.search(out_of_char, lexWord.lower())):
                lexicons.append(lexWord)

    for i, (image_tensors, labels_1) in enumerate(evaluation_loader):
        # print(i)
        if opt.debugFlag and i>2:
            break

        batch_size = image_tensors.size(0)
        #generate lexicons
        labels_2 = random.sample(lexicons, batch_size)


        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss_1, length_for_loss_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_for_loss_2, length_for_loss_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        
        start_time = time.time()
        images_recon_1, images_recon_2,_ = synthModel(image, text_for_loss_1, text_for_loss_2)

        #Save random reconstructed image and write its gt
        rIdx = random.randint(0,batch_size-1)
        try:
            save_image(tensor2im(image[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_input_'+labels_1[rIdx]+'.png'))
            save_image(tensor2im(images_recon_1[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_recon_'+labels_1[rIdx]+'.png'))
            save_image(tensor2im(images_recon_2[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_pair_'+labels_2[rIdx]+'.png'))
        except:
            print('Warning while saving validation image')
        
        
        if 'CTC' in opt.Prediction:
            preds_ocr = ocrModel(image, text_for_pred)
            preds_1 = ocrModel(images_recon_1, text_for_pred)
            preds_2 = ocrModel(images_recon_2, text_for_pred)

            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size_1 = torch.IntTensor([preds_1.size(1)] * batch_size)
            preds_size_2 = torch.IntTensor([preds_2.size(1)] * batch_size)

            # permute 'preds' to use CTCloss format
            ocrCost_ocr = ocrCriterion(preds_ocr.log_softmax(2).permute(1, 0, 2), text_for_loss_1, preds_size_1, length_for_loss_1)
            ocrCost_1 = ocrCriterion(preds_1.log_softmax(2).permute(1, 0, 2), text_for_loss_1, preds_size_1, length_for_loss_1)
            ocrCost_2 = ocrCriterion(preds_2.log_softmax(2).permute(1, 0, 2), text_for_loss_2, preds_size_2, length_for_loss_2)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index_ocr = preds_ocr.max(2)
            _, preds_index_1 = preds_1.max(2)
            _, preds_index_2 = preds_2.max(2)
            preds_str_ocr = converter.decode(preds_index_ocr.data, preds_size_1.data)
            preds_str_1 = converter.decode(preds_index_1.data, preds_size_1.data)
            preds_str_2 = converter.decode(preds_index_2.data, preds_size_2.data)
        
        else:
            
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        recCost = recCriterion(images_recon_1,image)

        infer_time += forward_time
        valid_loss_avg_ocr.add(ocrCost_ocr)
        valid_loss_avg.add(ocrCost_1+ocrCost_2+recCost)

        # calculate accuracy & confidence score
        preds_prob_ocr = F.softmax(preds_ocr, dim=2)
        preds_max_prob_ocr, _ = preds_prob_ocr.max(dim=2)

        preds_prob_1 = F.softmax(preds_1, dim=2)
        preds_max_prob_1, _ = preds_prob_1.max(dim=2)

        preds_prob_2 = F.softmax(preds_2, dim=2)
        preds_max_prob_2, _ = preds_prob_2.max(dim=2)

        confidence_score_list_ocr = []
        confidence_score_list_1 = []
        confidence_score_list_2 = []
        for gt_ocr, pred_ocr, pred_max_prob_ocr, gt_1, pred_1, pred_max_prob_1, gt_2, pred_2, pred_max_prob_2 in zip(labels_1, preds_str_ocr, preds_max_prob_ocr, labels_1, preds_str_1, preds_max_prob_1, labels_2, preds_str_2, preds_max_prob_2):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred_ocr == gt_ocr:
                n_correct_ocr += 1

            if pred_1 == gt_1:
                n_correct_1 += 1
            
            if pred_2 == gt_2:
                n_correct_2 += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt_1) == 0 or len(pred_1) == 0:
                norm_ED_1 += 0
            elif len(gt_1) > len(pred_1):
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(gt_1)
            else:
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(pred_1)

            # ICDAR2019 Normalized Edit Distance
            if len(gt_2) == 0 or len(pred_2) == 0:
                norm_ED_2 += 0
            elif len(gt_2) > len(pred_2):
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(gt_2)
            else:
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(pred_2)
            
            # ICDAR2019 Normalized Edit Distance
            if len(gt_ocr) == 0 or len(pred_ocr) == 0:
                norm_ED_ocr += 0
            elif len(gt_ocr) > len(pred_ocr):
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(gt_ocr)
            else:
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(pred_ocr)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score_ocr = pred_max_prob_ocr.cumprod(dim=0)[-1]
                confidence_score_1 = pred_max_prob_1.cumprod(dim=0)[-1]
                confidence_score_2 = pred_max_prob_2.cumprod(dim=0)[-1]
            except:
                confidence_score_ocr = 0
                confidence_score_1 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_2 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list_ocr.append(confidence_score_ocr)
            confidence_score_list_1.append(confidence_score_1)
            confidence_score_list_2.append(confidence_score_2)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy_ocr = n_correct_ocr / float(length_of_data) * 100
    norm_ED_ocr = norm_ED_ocr / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_1 = n_correct_1 / float(length_of_data) * 100
    norm_ED_1 = norm_ED_1 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_2 = n_correct_2 / float(length_of_data) * 100
    norm_ED_2 = norm_ED_2 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return [valid_loss_avg_ocr.val(), valid_loss_avg.val()], [accuracy_ocr,accuracy_1,accuracy_2], [norm_ED_ocr,norm_ED_1,norm_ED_2], [preds_str_ocr, preds_str_1,preds_str_2], [confidence_score_list_ocr,confidence_score_list_1,confidence_score_list_2], [labels_1,labels_1,labels_2], infer_time, length_of_data

def validation_synth_adv(iterCntr, synthModel, ocrModel, disModel, recCriterion, ocrCriterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr)), exist_ok=True)

    n_correct_ocr = 0
    norm_ED_ocr = 0

    n_correct_1 = 0
    norm_ED_1 = 0
    n_correct_2 = 0
    norm_ED_2 = 0

    length_of_data = 0
    infer_time = 0
    valid_loss_avg_ocr = Averager()
    valid_loss_avg = Averager()
    valid_loss_avg_dis = Averager()

    lexicons=[]
    out_of_char = f'[^{opt.character}]'
    #read lexicons file
    with open(opt.lexFile,'r') as lexF:
        for line in lexF:
            lexWord = line[:-1]
            if len(lexWord) <= opt.batch_max_length and not(re.search(out_of_char, lexWord.lower())):
                lexicons.append(lexWord)

    for i, (image_tensors_all, labels_1_all) in enumerate(evaluation_loader):
        # print(i)
        if opt.debugFlag and i>2:
            break
        
        disCnt = int(image_tensors_all.size(0)/2)
        image_tensors, image_tensors_real, labels_1 = image_tensors_all[:disCnt], image_tensors_all[disCnt:disCnt+disCnt], labels_1_all[:disCnt]

        batch_size = image_tensors.size(0)
        #generate lexicons
        labels_2 = random.sample(lexicons, batch_size)


        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        image_real = image_tensors_real.to(device)

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss_1, length_for_loss_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_for_loss_2, length_for_loss_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        
        start_time = time.time()
        images_recon_1, images_recon_2, _ = synthModel(image, text_for_loss_1, text_for_loss_2)

        #Save random reconstructed image and write its gt
        rIdx = random.randint(0,batch_size-1)
        try:
            save_image(tensor2im(image[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_input_'+labels_1[rIdx]+'.png'))
            save_image(tensor2im(images_recon_1[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_recon_'+labels_1[rIdx]+'.png'))
            save_image(tensor2im(images_recon_2[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_pair_'+labels_2[rIdx]+'.png'))
        except:
            print('Warning while saving validation image')
        
        
        if 'CTC' in opt.Prediction:
            preds_ocr = ocrModel(image, text_for_pred)
            preds_1 = ocrModel(images_recon_1, text_for_pred)
            preds_2 = ocrModel(images_recon_2, text_for_pred)

            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size_1 = torch.IntTensor([preds_1.size(1)] * batch_size)
            preds_size_2 = torch.IntTensor([preds_2.size(1)] * batch_size)

            # permute 'preds' to use CTCloss format
            ocrCost_ocr = ocrCriterion(preds_ocr.log_softmax(2).permute(1, 0, 2), text_for_loss_1, preds_size_1, length_for_loss_1)
            ocrCost_1 = ocrCriterion(preds_1.log_softmax(2).permute(1, 0, 2), text_for_loss_1, preds_size_1, length_for_loss_1)
            ocrCost_2 = ocrCriterion(preds_2.log_softmax(2).permute(1, 0, 2), text_for_loss_2, preds_size_2, length_for_loss_2)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index_ocr = preds_ocr.max(2)
            _, preds_index_1 = preds_1.max(2)
            _, preds_index_2 = preds_2.max(2)
            preds_str_ocr = converter.decode(preds_index_ocr.data, preds_size_1.data)
            preds_str_1 = converter.decode(preds_index_1.data, preds_size_1.data)
            preds_str_2 = converter.decode(preds_index_2.data, preds_size_2.data)

            disCost = 0.5*(disModel.module.calc_dis_loss(images_recon_1.detach(), image_real) + disModel.module.calc_dis_loss(images_recon_2.detach(), image))
            disGenCost = 0.5*(disModel.module.calc_gen_loss(images_recon_1)+disModel.module.calc_gen_loss(images_recon_2))
        else:
            
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        recCost = recCriterion(images_recon_1,image)

        infer_time += forward_time
        valid_loss_avg_ocr.add(ocrCost_ocr)
        valid_loss_avg.add(opt.ocrWeight*(0.5*(ocrCost_1+ocrCost_2))+opt.reconWeight*recCost+opt.disWeight*disGenCost)
        valid_loss_avg_dis.add(opt.disWeight*disCost)

        # calculate accuracy & confidence score
        preds_prob_ocr = F.softmax(preds_ocr, dim=2)
        preds_max_prob_ocr, _ = preds_prob_ocr.max(dim=2)

        preds_prob_1 = F.softmax(preds_1, dim=2)
        preds_max_prob_1, _ = preds_prob_1.max(dim=2)

        preds_prob_2 = F.softmax(preds_2, dim=2)
        preds_max_prob_2, _ = preds_prob_2.max(dim=2)

        confidence_score_list_ocr = []
        confidence_score_list_1 = []
        confidence_score_list_2 = []
        for gt_ocr, pred_ocr, pred_max_prob_ocr, gt_1, pred_1, pred_max_prob_1, gt_2, pred_2, pred_max_prob_2 in zip(labels_1, preds_str_ocr, preds_max_prob_ocr, labels_1, preds_str_1, preds_max_prob_1, labels_2, preds_str_2, preds_max_prob_2):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred_ocr == gt_ocr:
                n_correct_ocr += 1

            if pred_1 == gt_1:
                n_correct_1 += 1
            
            if pred_2 == gt_2:
                n_correct_2 += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt_1) == 0 or len(pred_1) == 0:
                norm_ED_1 += 0
            elif len(gt_1) > len(pred_1):
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(gt_1)
            else:
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(pred_1)

            # ICDAR2019 Normalized Edit Distance
            if len(gt_2) == 0 or len(pred_2) == 0:
                norm_ED_2 += 0
            elif len(gt_2) > len(pred_2):
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(gt_2)
            else:
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(pred_2)
            
            # ICDAR2019 Normalized Edit Distance
            if len(gt_ocr) == 0 or len(pred_ocr) == 0:
                norm_ED_ocr += 0
            elif len(gt_ocr) > len(pred_ocr):
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(gt_ocr)
            else:
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(pred_ocr)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score_ocr = pred_max_prob_ocr.cumprod(dim=0)[-1]
                confidence_score_1 = pred_max_prob_1.cumprod(dim=0)[-1]
                confidence_score_2 = pred_max_prob_2.cumprod(dim=0)[-1]
            except:
                confidence_score_ocr = 0
                confidence_score_1 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_2 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list_ocr.append(confidence_score_ocr)
            confidence_score_list_1.append(confidence_score_1)
            confidence_score_list_2.append(confidence_score_2)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy_ocr = n_correct_ocr / float(length_of_data) * 100
    norm_ED_ocr = norm_ED_ocr / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_1 = n_correct_1 / float(length_of_data) * 100
    norm_ED_1 = norm_ED_1 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_2 = n_correct_2 / float(length_of_data) * 100
    norm_ED_2 = norm_ED_2 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return [valid_loss_avg_ocr.val(), valid_loss_avg.val(), valid_loss_avg_dis.val()], [accuracy_ocr,accuracy_1,accuracy_2], [norm_ED_ocr,norm_ED_1,norm_ED_2], [preds_str_ocr, preds_str_1,preds_str_2], [confidence_score_list_ocr,confidence_score_list_1,confidence_score_list_2], [labels_1,labels_1,labels_2], infer_time, length_of_data

def validation_synth_lrw(iterCntr, synthModel, ocrModel, disModel, recCriterion, styleRecCriterion, ocrCriterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr)), exist_ok=True)

    n_correct_ocr = 0
    norm_ED_ocr = 0

    n_correct_1 = 0
    norm_ED_1 = 0
    n_correct_2 = 0
    norm_ED_2 = 0

    length_of_data = 0
    infer_time = 0
    valid_loss_avg_ocr = Averager()
    valid_loss_avg = Averager()
    valid_loss_avg_dis = Averager()

    lexicons=[]
    out_of_char = f'[^{opt.character}]'
    #read lexicons file
    with open(opt.lexFile,'r') as lexF:
        for line in lexF:
            lexWord = line[:-1]
            if len(lexWord) <= opt.batch_max_length and not(re.search(out_of_char, lexWord.lower())):
                lexicons.append(lexWord)

    for i, (image_tensors_all, labels_1_all) in enumerate(evaluation_loader):
        # print(i)
        if opt.debugFlag and i>2:
            break
        
        disCnt = int(image_tensors_all.size(0)/2)
        image_tensors, image_tensors_real, labels_1 = image_tensors_all[:disCnt], image_tensors_all[disCnt:disCnt+disCnt], labels_1_all[:disCnt]
        
        

        batch_size = image_tensors.size(0)
        #generate lexicons
        labels_2 = random.sample(lexicons, batch_size)


        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        image_real = image_tensors_real.to(device)

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss_1, length_for_loss_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_for_loss_2, length_for_loss_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        
        start_time = time.time()
        images_recon_1, images_recon_2, style = synthModel(image, text_for_loss_1, text_for_loss_2)

        #Save random reconstructed image and write its gt
        rIdx = random.randint(0,batch_size-1)
        try:
            save_image(tensor2im(image[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_input_'+labels_1[rIdx]+'.png'))
            save_image(tensor2im(images_recon_1[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_recon_'+labels_1[rIdx]+'.png'))
            save_image(tensor2im(images_recon_2[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_pair_'+labels_2[rIdx]+'.png'))
        except:
            print('Warning while saving validation image')
        
        
        if 'CTC' in opt.Prediction:
            preds_ocr = ocrModel(image, text_for_pred)
            preds_1 = ocrModel(images_recon_1, text_for_pred)
            preds_2 = ocrModel(images_recon_2, text_for_pred)

            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size_1 = torch.IntTensor([preds_1.size(1)] * batch_size)
            preds_size_2 = torch.IntTensor([preds_2.size(1)] * batch_size)

            # permute 'preds' to use CTCloss format
            ocrCost_ocr = ocrCriterion(preds_ocr.log_softmax(2).permute(1, 0, 2), text_for_loss_1, preds_size_1, length_for_loss_1)
            ocrCost_1 = ocrCriterion(preds_1.log_softmax(2).permute(1, 0, 2), text_for_loss_1, preds_size_1, length_for_loss_1)
            ocrCost_2 = ocrCriterion(preds_2.log_softmax(2).permute(1, 0, 2), text_for_loss_2, preds_size_2, length_for_loss_2)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index_ocr = preds_ocr.max(2)
            _, preds_index_1 = preds_1.max(2)
            _, preds_index_2 = preds_2.max(2)
            preds_str_ocr = converter.decode(preds_index_ocr.data, preds_size_1.data)
            preds_str_1 = converter.decode(preds_index_1.data, preds_size_1.data)
            preds_str_2 = converter.decode(preds_index_2.data, preds_size_2.data)

            disCost = 0.5*(disModel.module.calc_dis_loss(images_recon_1.detach(), image_real) + disModel.module.calc_dis_loss(images_recon_2.detach(), image))
            disGenCost = 0.5*(disModel.module.calc_gen_loss(images_recon_1)+disModel.module.calc_gen_loss(images_recon_2))
        else:
            
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        recCost = recCriterion(images_recon_1,image)
        styleRecCost = styleRecCriterion(synthModel(images_recon_2, None, None, styleFlag=True), style.detach())

        infer_time += forward_time
        valid_loss_avg_ocr.add(ocrCost_ocr)
        valid_loss_avg.add(opt.ocrWeight*(0.5*(ocrCost_1+ocrCost_2))+opt.reconWeight*recCost+opt.disWeight*disGenCost+opt.styleReconWeight*styleRecCost)
        valid_loss_avg_dis.add(opt.disWeight*disCost)

        # calculate accuracy & confidence score
        preds_prob_ocr = F.softmax(preds_ocr, dim=2)
        preds_max_prob_ocr, _ = preds_prob_ocr.max(dim=2)

        preds_prob_1 = F.softmax(preds_1, dim=2)
        preds_max_prob_1, _ = preds_prob_1.max(dim=2)

        preds_prob_2 = F.softmax(preds_2, dim=2)
        preds_max_prob_2, _ = preds_prob_2.max(dim=2)

        confidence_score_list_ocr = []
        confidence_score_list_1 = []
        confidence_score_list_2 = []
        for gt_ocr, pred_ocr, pred_max_prob_ocr, gt_1, pred_1, pred_max_prob_1, gt_2, pred_2, pred_max_prob_2 in zip(labels_1, preds_str_ocr, preds_max_prob_ocr, labels_1, preds_str_1, preds_max_prob_1, labels_2, preds_str_2, preds_max_prob_2):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred_ocr == gt_ocr:
                n_correct_ocr += 1

            if pred_1 == gt_1:
                n_correct_1 += 1
            
            if pred_2 == gt_2:
                n_correct_2 += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt_1) == 0 or len(pred_1) == 0:
                norm_ED_1 += 0
            elif len(gt_1) > len(pred_1):
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(gt_1)
            else:
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(pred_1)

            # ICDAR2019 Normalized Edit Distance
            if len(gt_2) == 0 or len(pred_2) == 0:
                norm_ED_2 += 0
            elif len(gt_2) > len(pred_2):
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(gt_2)
            else:
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(pred_2)
            
            # ICDAR2019 Normalized Edit Distance
            if len(gt_ocr) == 0 or len(pred_ocr) == 0:
                norm_ED_ocr += 0
            elif len(gt_ocr) > len(pred_ocr):
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(gt_ocr)
            else:
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(pred_ocr)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score_ocr = pred_max_prob_ocr.cumprod(dim=0)[-1]
                confidence_score_1 = pred_max_prob_1.cumprod(dim=0)[-1]
                confidence_score_2 = pred_max_prob_2.cumprod(dim=0)[-1]
            except:
                confidence_score_ocr = 0
                confidence_score_1 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_2 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list_ocr.append(confidence_score_ocr)
            confidence_score_list_1.append(confidence_score_1)
            confidence_score_list_2.append(confidence_score_2)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy_ocr = n_correct_ocr / float(length_of_data) * 100
    norm_ED_ocr = norm_ED_ocr / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_1 = n_correct_1 / float(length_of_data) * 100
    norm_ED_1 = norm_ED_1 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_2 = n_correct_2 / float(length_of_data) * 100
    norm_ED_2 = norm_ED_2 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return [valid_loss_avg_ocr.val(), valid_loss_avg.val(), valid_loss_avg_dis.val()], [accuracy_ocr,accuracy_1,accuracy_2], [norm_ED_ocr,norm_ED_1,norm_ED_2], [preds_str_ocr, preds_str_1,preds_str_2], [confidence_score_list_ocr,confidence_score_list_1,confidence_score_list_2], [labels_1,labels_1,labels_2], infer_time, length_of_data

def validation_synth_lrw_res(iterCntr, synthModel, ocrModel, disModel, recCriterion, styleRecCriterion, ocrCriterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr)), exist_ok=True)
    random.seed(1024)
    n_correct_ocr = 0
    norm_ED_ocr = 0

    n_correct_1 = 0
    norm_ED_1 = 0
    n_correct_2 = 0
    norm_ED_2 = 0

    length_of_data = 0
    infer_time = 0
    valid_loss_avg_ocr = Averager()
    valid_loss_avg = Averager()
    valid_loss_avg_dis = Averager()

    valid_loss_avg_ocrRecon_1 = Averager()
    valid_loss_avg_ocrRecon_2 = Averager()
    valid_loss_avg_gen = Averager()
    valid_loss_avg_imgRecon = Averager()
    valid_loss_avg_styRecon = Averager()

    lexicons=[]
    out_of_char = f'[^{opt.character}]'
    #read lexicons file
    with open(opt.lexFile,'r') as lexF:
        for line in lexF:
            lexWord = line[:-1]
            if opt.fixedString and len(lexWord)!=opt.batch_exact_length:
                continue
            if len(lexWord) <= opt.batch_max_length and not(re.search(out_of_char, lexWord.lower())) and len(lexWord) >= opt.batch_min_length:
                lexicons.append(lexWord)

    for i, (image_tensors_all, labels_1_all) in enumerate(evaluation_loader):
        # print(i)
        if opt.debugFlag and i>0:
            break
        
        disCnt = int(image_tensors_all.size(0)/2)
        
        image_tensors, image_tensors_real, labels_gt = image_tensors_all[:disCnt], image_tensors_all[disCnt:disCnt+disCnt], labels_1_all[:disCnt]
        image = image_tensors.to(device)
        image_real = image_tensors_real.to(device)
        batch_size = image_tensors.size(0)

        ##-----------------------------------##
        #generate text(labels) from ocr.forward
        if opt.ocrFixed:
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            
            if 'CTC' in opt.Prediction:
                preds = ocrModel(image, text_for_pred)
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
        else:
            labels_1 = labels_gt
        ##-----------------------------------##
        
        #generate lexicon labels
        labels_2 = random.sample(lexicons, batch_size)

        length_of_data = length_of_data + batch_size
        

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss_ocr, length_for_loss_ocr = converter.encode(labels_gt, batch_max_length=opt.batch_max_length)
        text_for_loss_1, length_for_loss_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_for_loss_2, length_for_loss_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        
        start_time = time.time()
        if image.shape[0] == 0:
            continue
        
        images_recon_1, images_recon_2, style = synthModel(image, text_for_loss_1, text_for_loss_2)
        
        

        # #Save random reconstructed image and write its gt
        # rIdx = random.randint(0,batch_size-1)
        # try:
        #     save_image(tensor2im(image[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_input_'+labels_gt[rIdx]+'.png'))
        #     save_image(tensor2im(images_recon_1[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_recon_'+labels_1[rIdx]+'.png'))
        #     save_image(tensor2im(images_recon_2[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_pair_'+labels_2[rIdx]+'.png'))
        # except:
        #     print('Warning while saving validation image')
        
        
        if 'CTC' in opt.Prediction:
            # if not opt.ocrFixed:
            #ocr evaluations with orig image
            preds_ocr = ocrModel(image, text_for_pred)
            preds_size_ocr = torch.IntTensor([preds_ocr.size(1)] * batch_size)
            ocrCost_ocr = ocrCriterion(preds_ocr.log_softmax(2).permute(1, 0, 2), text_for_loss_ocr, preds_size_ocr, length_for_loss_ocr)
            _, preds_index_ocr = preds_ocr.max(2)
            preds_str_ocr = converter.decode(preds_index_ocr.data, preds_size_ocr.data)
            
            #content loss for reconstructed images
            # permute 'preds' to use CTCloss format
            preds_1 = ocrModel(images_recon_1, text_for_pred)
            preds_size_1 = torch.IntTensor([preds_1.size(1)] * batch_size)
            ocrCost_1 = ocrCriterion(preds_1.log_softmax(2).permute(1, 0, 2), text_for_loss_1, preds_size_1, length_for_loss_1)
            _, preds_index_1 = preds_1.max(2)
            preds_str_1 = converter.decode(preds_index_1.data, preds_size_1.data)
            
            preds_2 = ocrModel(images_recon_2, text_for_pred)
            preds_size_2 = torch.IntTensor([preds_2.size(1)] * batch_size)
            ocrCost_2 = ocrCriterion(preds_2.log_softmax(2).permute(1, 0, 2), text_for_loss_2, preds_size_2, length_for_loss_2)
            _, preds_index_2 = preds_2.max(2)
            preds_str_2 = converter.decode(preds_index_2.data, preds_size_2.data)

        else:
            # if not opt.ocrFixed:
            #ocr evaluations with orig image
            preds_ocr = ocrModel(image, text_for_pred, is_train=False)
            
            preds_ocr = preds_ocr[:, :text_for_loss_ocr.shape[1] - 1, :]
            target_ocr = text_for_loss_ocr[:, 1:]  # without [GO] Symbol
            ocrCost_ocr = ocrCriterion(preds_ocr.contiguous().view(-1, preds_ocr.shape[-1]), target_ocr.contiguous().view(-1))
            _, preds_index = preds_ocr.max(2)
            preds_str_ocr = converter.decode(preds_index, length_for_pred)
            # labels_1 = converter.decode(text_for_loss_1[:, 1:], length_for_loss_1)
            # else:
            #     ocrCost_ocr = torch.tensor(0.0)
            
            #ocr evaluations with orig image
            preds_1 = ocrModel(images_recon_1, text_for_pred, is_train=False)
            preds_1 = preds_1[:, :text_for_loss_1.shape[1] - 1, :]
            target_1 = text_for_loss_1[:, 1:]  # without [GO] Symbol
            ocrCost_1 = ocrCriterion(preds_1.contiguous().view(-1, preds_1.shape[-1]), target_1.contiguous().view(-1))
            _, preds_index_1 = preds_1.max(2)
            preds_str_1 = converter.decode(preds_index_1, length_for_pred)

            preds_2 = ocrModel(images_recon_2, text_for_pred, is_train=False)
            preds_2 = preds_2[:, :text_for_loss_2.shape[1] - 1, :]
            target_2 = text_for_loss_2[:, 1:]  # without [GO] Symbol
            ocrCost_2 = ocrCriterion(preds_2.contiguous().view(-1, preds_2.shape[-1]), target_2.contiguous().view(-1))
            _, preds_index_2 = preds_2.max(2)
            preds_str_2 = converter.decode(preds_index_2, length_for_pred)

        forward_time = time.time() - start_time

        if disModel == None:
            disCost = torch.tensor(0.0)
            disGenCost = torch.tensor(0.0)
        else:
            if opt.gan_type == 'wgan':
                disCost = torch.tensor(0.0)
            else:
                disCost = 0.5*(disModel.module.calc_dis_loss(images_recon_1.detach(), image_real) + disModel.module.calc_dis_loss(images_recon_2.detach(), image))
            disGenCost = 0.5*(disModel.module.calc_gen_loss(images_recon_1)+disModel.module.calc_gen_loss(images_recon_2))
        recCost = recCriterion(images_recon_1,image)
        
        if opt.styleReconWeight == 0.0:
            styleRecCost = torch.tensor(0.0)
        else:
            styleRecCost = styleRecCriterion(synthModel(images_recon_2, None, None, styleFlag=True), style)

        infer_time += forward_time
        valid_loss_avg_ocr.add(ocrCost_ocr)
        valid_loss_avg.add(opt.ocrWeight*(0.5*(ocrCost_1+ocrCost_2))+opt.reconWeight*recCost+opt.disWeight*disGenCost+opt.styleReconWeight*styleRecCost)
        valid_loss_avg_dis.add(opt.disWeight*disCost)
        
        #fine grained losses
        valid_loss_avg_ocrRecon_1.add(opt.ocrWeight*(0.5*(ocrCost_1)))
        valid_loss_avg_ocrRecon_2.add(opt.ocrWeight*(0.5*(ocrCost_2)))
        valid_loss_avg_gen.add(opt.disWeight*disGenCost)
        valid_loss_avg_imgRecon.add(opt.reconWeight*recCost)
        valid_loss_avg_styRecon.add(opt.styleReconWeight*styleRecCost)

        # if not opt.ocrFixed:
        # calculate accuracy & confidence score
        preds_prob_ocr = F.softmax(preds_ocr, dim=2)
        preds_max_prob_ocr, _ = preds_prob_ocr.max(dim=2)

        preds_prob_1 = F.softmax(preds_1, dim=2)
        preds_max_prob_1, _ = preds_prob_1.max(dim=2)

        preds_prob_2 = F.softmax(preds_2, dim=2)
        preds_max_prob_2, _ = preds_prob_2.max(dim=2)

        confidence_score_list_ocr = []
        confidence_score_list_1 = []
        confidence_score_list_2 = []

        # zCntr=0
        for gt_ocr, pred_ocr, pred_max_prob_ocr, gt_1, pred_1, pred_max_prob_1, gt_2, pred_2, pred_max_prob_2 in zip(labels_gt, preds_str_ocr, preds_max_prob_ocr, labels_1, preds_str_1, preds_max_prob_1, labels_2, preds_str_2, preds_max_prob_2):
            if 'Attn' in opt.Prediction:
                # if not opt.ocrFixed:
                
                # gt_ocr = gt_ocr[:gt_ocr.find('[s]')]
                pred_EOS = pred_ocr.find('[s]')
                pred_ocr = pred_ocr[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob_ocr = pred_max_prob_ocr[:pred_EOS]
                
                # gt_1 = gt_1[:gt_1.find('[s]')]
                pred_EOS = pred_1.find('[s]')
                pred_1 = pred_1[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob_1 = pred_max_prob_1[:pred_EOS]

                # gt_2 = gt_2[:gt_2.find('[s]')]
                pred_EOS = pred_2.find('[s]')
                pred_2 = pred_2[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob_2 = pred_max_prob_2[:pred_EOS]

            # # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            # if opt.sensitive and opt.data_filtering_off:
            #     pred = pred.lower()
            #     gt = gt.lower()
            #     alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            #     out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            #     pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
            #     gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred_ocr == gt_ocr:
                n_correct_ocr += 1
            # else:
            #     n_correct_ocr=0

            if pred_1 == gt_1:
                n_correct_1 += 1
            
            if pred_2 == gt_2:
                n_correct_2 += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt_1) == 0 or len(pred_1) == 0:
                norm_ED_1 += 0
            elif len(gt_1) > len(pred_1):
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(gt_1)
            else:
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(pred_1)

            # ICDAR2019 Normalized Edit Distance
            if len(gt_2) == 0 or len(pred_2) == 0:
                norm_ED_2 += 0
            elif len(gt_2) > len(pred_2):
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(gt_2)
            else:
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(pred_2)
            
            # if not opt.ocrFixed:
            # ICDAR2019 Normalized Edit Distance
            if len(gt_ocr) == 0 or len(pred_ocr) == 0:
                norm_ED_ocr += 0
            elif len(gt_ocr) > len(pred_ocr):
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(gt_ocr)
            else:
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(pred_ocr)
            # else:
            #     norm_ED_ocr=0

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                # if not opt.ocrFixed:
                confidence_score_ocr = pred_max_prob_ocr.cumprod(dim=0)[-1]
                # else:
                #     confidence_score_ocr = 1.0
                confidence_score_1 = pred_max_prob_1.cumprod(dim=0)[-1]
                confidence_score_2 = pred_max_prob_2.cumprod(dim=0)[-1]
            except:
                confidence_score_ocr = 0
                confidence_score_1 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_2 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list_ocr.append(confidence_score_ocr)
            confidence_score_list_1.append(confidence_score_1)
            confidence_score_list_2.append(confidence_score_2)
            # print(pred, gt, pred==gt, confidence_score)
            
            # zCntr+=1
        #Save random reconstructed image and write its gt
        if opt.testFlag:
            randomSaveIdx = list(range(0,batch_size))
        else:
            randomSaveIdx = [random.randint(0,batch_size-1)]
        for rIdx in randomSaveIdx:
            if 'Attn' in opt.Prediction:
                r_pred_EOS = preds_str_ocr[rIdx].find('[s]')
                r_pred_ocr = preds_str_ocr[rIdx][:r_pred_EOS]

                r_pred_1_EOS = preds_str_1[rIdx].find('[s]')
                r_pred_1 = preds_str_1[rIdx][:r_pred_1_EOS]

                r_pred_2_EOS = preds_str_2[rIdx].find('[s]')
                r_pred_2 = preds_str_2[rIdx][:r_pred_2_EOS]
            else:
                r_pred_ocr = preds_str_ocr[rIdx]
                r_pred_1 = preds_str_1[rIdx]
                r_pred_2 = preds_str_2[rIdx]
            
            try:
                save_image(tensor2im(image[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_'+str(rIdx)+'_'+'_input_'+labels_gt[rIdx]+'_'+r_pred_ocr+'.png'))
                save_image(tensor2im(images_recon_1[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_'+str(rIdx)+'_'+'_recon_'+labels_1[rIdx]+'_'+r_pred_1+'.png'))
                save_image(tensor2im(images_recon_2[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_'+str(rIdx)+'_'+'_pair_'+labels_2[rIdx]+'_'+r_pred_2+'.png'))
            except:
                print('Warning while saving validation image')

    accuracy_ocr = n_correct_ocr / float(length_of_data) * 100
    norm_ED_ocr = norm_ED_ocr / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_1 = n_correct_1 / float(length_of_data) * 100
    norm_ED_1 = norm_ED_1 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_2 = n_correct_2 / float(length_of_data) * 100
    norm_ED_2 = norm_ED_2 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance
    
    random.seed()

    return [valid_loss_avg_ocr.val(), valid_loss_avg.val(), valid_loss_avg_dis.val(), valid_loss_avg_ocrRecon_1.val(),valid_loss_avg_ocrRecon_2.val(), valid_loss_avg_gen.val(), valid_loss_avg_imgRecon.val(), valid_loss_avg_styRecon.val()], [accuracy_ocr,accuracy_1,accuracy_2], [norm_ED_ocr,norm_ED_1,norm_ED_2], [preds_str_ocr, preds_str_1,preds_str_2], [confidence_score_list_ocr,confidence_score_list_1,confidence_score_list_2], [labels_gt,labels_1,labels_2], infer_time, length_of_data

def test_synth_lrw_res(iterCntr, synthModel, ocrModel, disModel, recCriterion, styleRecCriterion, ocrCriterion, evaluation_loader, converter, opt):
    
    
    """ validation or evaluation """
    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr)), exist_ok=True)
    random.seed(1024)
    n_correct_ocr = 0
    norm_ED_ocr = 0

    n_correct_1 = 0
    norm_ED_1 = 0
    n_correct_2 = 0
    norm_ED_2 = 0

    length_of_data = 0
    infer_time = 0
    valid_loss_avg_ocr = Averager()
    valid_loss_avg = Averager()
    valid_loss_avg_dis = Averager()

    valid_loss_avg_ocrRecon_1 = Averager()
    valid_loss_avg_ocrRecon_2 = Averager()
    valid_loss_avg_gen = Averager()
    valid_loss_avg_imgRecon = Averager()
    valid_loss_avg_styRecon = Averager()

    lexicons=[]
    out_of_char = f'[^{opt.character}]'
    #read lexicons file
    with open(opt.lexFile,'r') as lexF:
        for line in lexF:
            lexWord = line[:-1]
            if opt.fixedString and len(lexWord)!=opt.batch_exact_length:
                continue
            if len(lexWord) <= opt.batch_max_length and not(re.search(out_of_char, lexWord.lower())) and len(lexWord) >= opt.batch_min_length:
                lexicons.append(lexWord)

    
    for i, (image_tensors, labels_gt) in enumerate(evaluation_loader):
        # print(i)
        
        if opt.debugFlag and i>0:
            break
        
        # disCnt = int(image_tensors_all.size(0)/2)
        
        # image_tensors, image_tensors_real, labels_gt = image_tensors_all[:disCnt], image_tensors_all[disCnt:disCnt+disCnt], labels_1_all[:disCnt]
        image = image_tensors.to(device)
        # image_real = image_tensors_real.to(device)
        batch_size = image_tensors.size(0)

        ##-----------------------------------##
        #generate text(labels) from ocr.forward
        if opt.ocrFixed:
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            
            if 'CTC' in opt.Prediction:
                preds = ocrModel(image, text_for_pred)
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
        else:
            labels_1 = labels_gt
        ##-----------------------------------##
        
        #generate lexicon labels
        labels_2 = random.sample(lexicons, batch_size)
        # labels_2 = ['gtp']*batch_size
        length_of_data = length_of_data + batch_size
        

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss_ocr, length_for_loss_ocr = converter.encode(labels_gt, batch_max_length=opt.batch_max_length)
        text_for_loss_1, length_for_loss_1 = converter.encode(labels_1, batch_max_length=opt.batch_max_length)
        text_for_loss_2, length_for_loss_2 = converter.encode(labels_2, batch_max_length=opt.batch_max_length)
        
        start_time = time.time()
        if image.shape[0] == 0:
            continue
        
        images_recon_1, images_recon_2, style = synthModel(image, text_for_loss_1, text_for_loss_2)
        
        

        # #Save random reconstructed image and write its gt
        # rIdx = random.randint(0,batch_size-1)
        # try:
        #     save_image(tensor2im(image[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_input_'+labels_gt[rIdx]+'.png'))
        #     save_image(tensor2im(images_recon_1[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_recon_'+labels_1[rIdx]+'.png'))
        #     save_image(tensor2im(images_recon_2[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_pair_'+labels_2[rIdx]+'.png'))
        # except:
        #     print('Warning while saving validation image')
        
        
        if 'CTC' in opt.Prediction:
            # if not opt.ocrFixed:
            #ocr evaluations with orig image
            preds_ocr = ocrModel(image, text_for_pred)
            preds_size_ocr = torch.IntTensor([preds_ocr.size(1)] * batch_size)
            ocrCost_ocr = ocrCriterion(preds_ocr.log_softmax(2).permute(1, 0, 2), text_for_loss_ocr, preds_size_ocr, length_for_loss_ocr)
            _, preds_index_ocr = preds_ocr.max(2)
            preds_str_ocr = converter.decode(preds_index_ocr.data, preds_size_ocr.data)
            
            #content loss for reconstructed images
            # permute 'preds' to use CTCloss format
            preds_1 = ocrModel(images_recon_1, text_for_pred)
            preds_size_1 = torch.IntTensor([preds_1.size(1)] * batch_size)
            ocrCost_1 = ocrCriterion(preds_1.log_softmax(2).permute(1, 0, 2), text_for_loss_1, preds_size_1, length_for_loss_1)
            _, preds_index_1 = preds_1.max(2)
            preds_str_1 = converter.decode(preds_index_1.data, preds_size_1.data)
            
            preds_2 = ocrModel(images_recon_2, text_for_pred)
            preds_size_2 = torch.IntTensor([preds_2.size(1)] * batch_size)
            ocrCost_2 = ocrCriterion(preds_2.log_softmax(2).permute(1, 0, 2), text_for_loss_2, preds_size_2, length_for_loss_2)
            _, preds_index_2 = preds_2.max(2)
            preds_str_2 = converter.decode(preds_index_2.data, preds_size_2.data)

        else:
            # if not opt.ocrFixed:
            #ocr evaluations with orig image
            preds_ocr = ocrModel(image, text_for_pred, is_train=False)
            
            preds_ocr = preds_ocr[:, :text_for_loss_ocr.shape[1] - 1, :]
            target_ocr = text_for_loss_ocr[:, 1:]  # without [GO] Symbol
            ocrCost_ocr = ocrCriterion(preds_ocr.contiguous().view(-1, preds_ocr.shape[-1]), target_ocr.contiguous().view(-1))
            _, preds_index = preds_ocr.max(2)
            preds_str_ocr = converter.decode(preds_index, length_for_pred)
            # labels_1 = converter.decode(text_for_loss_1[:, 1:], length_for_loss_1)
            # else:
            #     ocrCost_ocr = torch.tensor(0.0)
            
            #ocr evaluations with orig image
            preds_1 = ocrModel(images_recon_1, text_for_pred, is_train=False)
            preds_1 = preds_1[:, :text_for_loss_1.shape[1] - 1, :]
            target_1 = text_for_loss_1[:, 1:]  # without [GO] Symbol
            ocrCost_1 = ocrCriterion(preds_1.contiguous().view(-1, preds_1.shape[-1]), target_1.contiguous().view(-1))
            _, preds_index_1 = preds_1.max(2)
            preds_str_1 = converter.decode(preds_index_1, length_for_pred)

            preds_2 = ocrModel(images_recon_2, text_for_pred, is_train=False)
            preds_2 = preds_2[:, :text_for_loss_2.shape[1] - 1, :]
            target_2 = text_for_loss_2[:, 1:]  # without [GO] Symbol
            ocrCost_2 = ocrCriterion(preds_2.contiguous().view(-1, preds_2.shape[-1]), target_2.contiguous().view(-1))
            _, preds_index_2 = preds_2.max(2)
            preds_str_2 = converter.decode(preds_index_2, length_for_pred)

        forward_time = time.time() - start_time

        if disModel == None:
            disCost = torch.tensor(0.0)
            disGenCost = torch.tensor(0.0)
        else:
            if opt.gan_type == 'wgan':
                disCost = torch.tensor(0.0)
            else:
                disCost = 0.5*(disModel.module.calc_dis_loss(images_recon_1.detach(), image_real) + disModel.module.calc_dis_loss(images_recon_2.detach(), image))
            disGenCost = 0.5*(disModel.module.calc_gen_loss(images_recon_1)+disModel.module.calc_gen_loss(images_recon_2))
        recCost = recCriterion(images_recon_1,image)
        
        if opt.styleReconWeight == 0.0:
            styleRecCost = torch.tensor(0.0)
        else:
            styleRecCost = styleRecCriterion(synthModel(images_recon_2, None, None, styleFlag=True), style)

        infer_time += forward_time
        valid_loss_avg_ocr.add(ocrCost_ocr)
        valid_loss_avg.add(opt.ocrWeight*(0.5*(ocrCost_1+ocrCost_2))+opt.reconWeight*recCost+opt.disWeight*disGenCost+opt.styleReconWeight*styleRecCost)
        valid_loss_avg_dis.add(opt.disWeight*disCost)
        
        #fine grained losses
        valid_loss_avg_ocrRecon_1.add(opt.ocrWeight*(0.5*(ocrCost_1)))
        valid_loss_avg_ocrRecon_2.add(opt.ocrWeight*(0.5*(ocrCost_2)))
        valid_loss_avg_gen.add(opt.disWeight*disGenCost)
        valid_loss_avg_imgRecon.add(opt.reconWeight*recCost)
        valid_loss_avg_styRecon.add(opt.styleReconWeight*styleRecCost)

        # if not opt.ocrFixed:
        # calculate accuracy & confidence score
        preds_prob_ocr = F.softmax(preds_ocr, dim=2)
        preds_max_prob_ocr, _ = preds_prob_ocr.max(dim=2)

        preds_prob_1 = F.softmax(preds_1, dim=2)
        preds_max_prob_1, _ = preds_prob_1.max(dim=2)

        preds_prob_2 = F.softmax(preds_2, dim=2)
        preds_max_prob_2, _ = preds_prob_2.max(dim=2)

        confidence_score_list_ocr = []
        confidence_score_list_1 = []
        confidence_score_list_2 = []

        # zCntr=0
        for gt_ocr, pred_ocr, pred_max_prob_ocr, gt_1, pred_1, pred_max_prob_1, gt_2, pred_2, pred_max_prob_2 in zip(labels_gt, preds_str_ocr, preds_max_prob_ocr, labels_1, preds_str_1, preds_max_prob_1, labels_2, preds_str_2, preds_max_prob_2):
            if 'Attn' in opt.Prediction:
                # if not opt.ocrFixed:
                
                # gt_ocr = gt_ocr[:gt_ocr.find('[s]')]
                pred_EOS = pred_ocr.find('[s]')
                pred_ocr = pred_ocr[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob_ocr = pred_max_prob_ocr[:pred_EOS]
                
                # gt_1 = gt_1[:gt_1.find('[s]')]
                pred_EOS = pred_1.find('[s]')
                pred_1 = pred_1[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob_1 = pred_max_prob_1[:pred_EOS]

                # gt_2 = gt_2[:gt_2.find('[s]')]
                pred_EOS = pred_2.find('[s]')
                pred_2 = pred_2[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob_2 = pred_max_prob_2[:pred_EOS]

            # # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            # if opt.sensitive and opt.data_filtering_off:
            #     pred = pred.lower()
            #     gt = gt.lower()
            #     alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            #     out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            #     pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
            #     gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred_ocr == gt_ocr:
                n_correct_ocr += 1
            # else:
            #     n_correct_ocr=0

            if pred_1 == gt_1:
                n_correct_1 += 1
            
            if pred_2 == gt_2:
                n_correct_2 += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt_1) == 0 or len(pred_1) == 0:
                norm_ED_1 += 0
            elif len(gt_1) > len(pred_1):
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(gt_1)
            else:
                norm_ED_1 += 1 - edit_distance(pred_1, gt_1) / len(pred_1)

            # ICDAR2019 Normalized Edit Distance
            if len(gt_2) == 0 or len(pred_2) == 0:
                norm_ED_2 += 0
            elif len(gt_2) > len(pred_2):
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(gt_2)
            else:
                norm_ED_2 += 1 - edit_distance(pred_2, gt_2) / len(pred_2)
            
            # if not opt.ocrFixed:
            # ICDAR2019 Normalized Edit Distance
            if len(gt_ocr) == 0 or len(pred_ocr) == 0:
                norm_ED_ocr += 0
            elif len(gt_ocr) > len(pred_ocr):
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(gt_ocr)
            else:
                norm_ED_ocr += 1 - edit_distance(pred_ocr, gt_ocr) / len(pred_ocr)
            # else:
            #     norm_ED_ocr=0

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                # if not opt.ocrFixed:
                confidence_score_ocr = pred_max_prob_ocr.cumprod(dim=0)[-1]
                # else:
                #     confidence_score_ocr = 1.0
                confidence_score_1 = pred_max_prob_1.cumprod(dim=0)[-1]
                confidence_score_2 = pred_max_prob_2.cumprod(dim=0)[-1]
            except:
                confidence_score_ocr = 0
                confidence_score_1 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_2 = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list_ocr.append(confidence_score_ocr)
            confidence_score_list_1.append(confidence_score_1)
            confidence_score_list_2.append(confidence_score_2)
            # print(pred, gt, pred==gt, confidence_score)
            
            # zCntr+=1
        #Save random reconstructed image and write its gt
        if opt.testFlag:
            randomSaveIdx = list(range(0,batch_size))
        else:
            randomSaveIdx = [random.randint(0,batch_size-1)]
        for rIdx in randomSaveIdx:
            if 'Attn' in opt.Prediction:
                r_pred_EOS = preds_str_ocr[rIdx].find('[s]')
                r_pred_ocr = preds_str_ocr[rIdx][:r_pred_EOS]

                r_pred_1_EOS = preds_str_1[rIdx].find('[s]')
                r_pred_1 = preds_str_1[rIdx][:r_pred_1_EOS]

                r_pred_2_EOS = preds_str_2[rIdx].find('[s]')
                r_pred_2 = preds_str_2[rIdx][:r_pred_2_EOS]
            else:
                r_pred_ocr = preds_str_ocr[rIdx]
                r_pred_1 = preds_str_1[rIdx]
                r_pred_2 = preds_str_2[rIdx]
            
            try:
                save_image(tensor2im(image[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_'+str(rIdx)+'_'+'_input_'+labels_gt[rIdx]+'_'+r_pred_ocr+'.png'))
                save_image(tensor2im(images_recon_1[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_'+str(rIdx)+'_'+'_recon_'+labels_1[rIdx]+'_'+r_pred_1+'.png'))
                save_image(tensor2im(images_recon_2[rIdx]),os.path.join(opt.exp_dir,opt.exp_name,'valImages',str(iterCntr),str(i)+'_'+str(rIdx)+'_'+'_pair_'+labels_2[rIdx]+'_'+r_pred_2+'.png'))
            except:
                print('Warning while saving validation image')

    accuracy_ocr = n_correct_ocr / float(length_of_data) * 100
    norm_ED_ocr = norm_ED_ocr / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_1 = n_correct_1 / float(length_of_data) * 100
    norm_ED_1 = norm_ED_1 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    accuracy_2 = n_correct_2 / float(length_of_data) * 100
    norm_ED_2 = norm_ED_2 / float(length_of_data)  # ICDAR2019 Normalized Edit Distance
    
    random.seed()

    return [valid_loss_avg_ocr.val(), valid_loss_avg.val(), valid_loss_avg_dis.val(), valid_loss_avg_ocrRecon_1.val(),valid_loss_avg_ocrRecon_2.val(), valid_loss_avg_gen.val(), valid_loss_avg_imgRecon.val(), valid_loss_avg_styRecon.val()], [accuracy_ocr,accuracy_1,accuracy_2], [norm_ED_ocr,norm_ED_1,norm_ED_2], [preds_str_ocr, preds_str_1,preds_str_2], [confidence_score_list_ocr,confidence_score_list_1,confidence_score_list_2], [labels_gt,labels_1,labels_2], infer_time, length_of_data

def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    
    model = AdaINGenV4(opt)
    ocrModel = Model(opt)

    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    
    model = torch.nn.DataParallel(model).to(device)
    ocrModel = torch.nn.DataParallel(ocrModel).to(device)

    # load model
    print('loading pretrained ocr model from %s' % opt.saved_ocr_model)
    ocrModel.load_state_dict(torch.load(opt.saved_ocr_model, map_location=device))

    print('loading pretrained synth model from %s' % opt.saved_synth_model)
    model.load_state_dict(torch.load(opt.saved_synth_model, map_location=device))

    # opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name), exist_ok=True)
    os.makedirs(os.path.join(opt.exp_dir,opt.exp_name,'evalImages'), exist_ok=True)

    print(model)
    print(ocrModel)

    


    """ keep evaluation model and result logs """
    # os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    # os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        ocrCriterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        ocrCriterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    recCriterion = torch.nn.L1Loss()
    styleRecCriterion = torch.nn.L1Loss()

    """ evaluation """
    model.eval()
    ocrModel.eval()

    with torch.no_grad():
        if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            benchmark_all_eval(model, ocrModel, recCriterion, styleRecCriterion, ocrCriterion, converter, opt)
        else:
            # log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            test_synth_lrw_res(-1,model, ocrModel, None, recCriterion, styleRecCriterion, ocrCriterion, evaluation_loader, converter, opt)
            # log.write(eval_data_log)
            # print(f'{accuracy_by_best_model:0.3f}')
            # log.write(f'{accuracy_by_best_model:0.3f}\n')
            # log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp06/', help='Where to store logs and models')
    parser.add_argument('--exp_name', default='debug', help='Where to store logs and models')
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_ocr_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_synth_model', default='', help="path to model to continue training")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--batch_min_length', type=int, default=1, help='minimum-label-length')
    parser.add_argument('--fixedString', action='store_true', help='use fixed length data')
    parser.add_argument('--batch_exact_length', type=int, default=5, help='exact-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--ocr_imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--ocr_imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--lexFile', default='/checkpoint/pkrishnan/datasets/vocab/english-words.txt', help='unqiue words in language')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
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

    parser.add_argument('--debugFlag', action='store_true', help='for debugging')
    parser.add_argument('--testFlag', action='store_true', help='for testing')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
