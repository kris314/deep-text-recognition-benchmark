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
import html_visual as html

from dataset import fontDataset, AlignFontCollate
from utils import Averager

from model import VGGFontModel

import tflib as lib
import tflib.plot

sys.path.append('/private/home/pkrishnan/codes/st-scribe/stylegan2-pytorch/')


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


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def train(opt):
    lib.print_model_settings(locals().copy())
    
    
    # train_transform =  transforms.Compose([
    #     # transforms.RandomResizedCrop(input_size),
    #     transforms.Resize((opt.imgH, opt.imgW)),
    #     transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=None, shear=None, resample=False, fillcolor=0),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    
    # val_transform = transforms.Compose([
    #     transforms.Resize((opt.imgH, opt.imgW)),
    #     # transforms.CenterCrop(input_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    AlignFontCollateObj = AlignFontCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    train_dataset = fontDataset(imgDir=opt.train_img_dir, annFile=opt.train_ann_file, transform=None, numClasses=opt.numClasses)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, 
        shuffle=False,  # 'True' to check training progress with validation function.
        sampler=data_sampler(train_dataset, shuffle=True, distributed=opt.distributed),
        num_workers=int(opt.workers),
        collate_fn=AlignFontCollateObj, pin_memory=True, drop_last=False)
    # numClasses = len(train_dataset.Idx2F)
    numClasses = np.unique(train_dataset.fontIdx).size
    
    train_loader = sample_data(train_loader)
    print('-' * 80)
    numTrainSamples = len(train_dataset)

    # valid_dataset = LmdbStyleDataset(root=opt.valid_data, opt=opt)
    valid_dataset = fontDataset(imgDir=opt.train_img_dir, annFile=opt.val_ann_file, transform=None, F2Idx=train_dataset.F2Idx, Idx2F=train_dataset.Idx2F, numClasses=opt.numClasses)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, 
        shuffle=False,  # 'True' to check training progress with validation function.
        sampler=data_sampler(valid_dataset, shuffle=False, distributed=opt.distributed),
        num_workers=int(opt.workers),
        collate_fn=AlignFontCollateObj, pin_memory=True, drop_last=False)
    numTestSamples = len(valid_dataset)

    print('numClasses', numClasses)
    print('numTrainSamples', numTrainSamples)
    print('numTestSamples', numTestSamples)
    
    vggFontModel = VGGFontModel(models.vgg19(pretrained=opt.preTrained), numClasses).to(device)
    for name, param in vggFontModel.classifier.named_parameters():
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
    
    if opt.optim == "sgd":
        print('SGD optimizer')
        optimizer = optim.SGD(
            vggFontModel.parameters(),
            lr=opt.lr ,
            momentum=0.9)
    elif opt.optim == "adam":
        print('Adam optimizer')
        optimizer = optim.Adam(
            vggFontModel.parameters(),
            lr=opt.lr)
    
    #get schedulers
    scheduler = get_scheduler(optimizer,opt)

    criterion = torch.nn.CrossEntropyLoss()

    if opt.modelFolderFlag:
        if len(glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_vggFont.pth")))>0:
            opt.saved_font_model = glob.glob(os.path.join(opt.exp_dir,opt.exp_name,"iter_*_vggFont.pth"))[-1]

    ## Loading pre-trained files
    if opt.saved_font_model != '' and opt.saved_font_model != 'None':
        print(f'loading pretrained synth model from {opt.saved_font_model}')
        checkpoint = torch.load(opt.saved_font_model, map_location=lambda storage, loc: storage)
        
        # if checkpoint['vggFontModel']['classifier.6.weight'].shape[0] != numClasses:
        #     own_state = vggFontModel.state_dict()
        #     for name, param in checkpoint['vggFontModel'].items():
        #         if name in ['classifier.6.weight', 'classifier.6.bias']:
        #             continue
        #         if isinstance(param, torch.nn.Parameter):
        #             # backwards compatibility for serialized parameters
        #             param = param.data
        #         own_state[name].copy_(param)
        # else:
        vggFontModel.load_state_dict(checkpoint['vggFontModel'])
        
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    # print('Model Initialization')
    #   
    # print('Loaded checkpoint')

    if  opt.distributed:
        vggFontModel = torch.nn.parallel.DistributedDataParallel(
            vggFontModel,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        vggFontModel.train()

    # print('Loaded distributed')

    if opt.distributed:
        vggFontModel_module = vggFontModel.module        
    else:
        vggFontModel_module = vggFontModel


    # print('Loading module')

    # loss averager
    loss_train = Averager()
    loss_val = Averager()
    train_acc = Averager()
    val_acc = Averager()
    train_acc_5 = Averager()
    val_acc_5 = Averager()
    val_acc_10 = Averager()

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
    
    if opt.saved_font_model != '' and opt.saved_font_model != 'None':
        try:
            start_iter = int(opt.saved_font_model.split('_')[-2].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    
    
    iteration = start_iter

    cntr=0
    # trainCorrect=0
    # tCntr=0
    while(True):
        # print(cntr)
        # train part
        
        start_time = time.time()
        if not opt.testFlag:
            
            image_input_tensors, labels_gt  = next(train_loader)
            image_input_tensors = image_input_tensors.to(device)
            labels_gt = labels_gt.view(-1).to(device)
            preds = vggFontModel(image_input_tensors)
            
            loss = criterion(preds, labels_gt)
            
            vggFontModel.zero_grad()
            loss.backward()
            optimizer.step()

            # _, preds_max = preds.max(dim=1)
            # trainCorrect += (preds_max == labels_gt).sum()
            # tCntr+=preds_max.shape[0]
            
            acc1, acc5 = getNumCorrect(preds, labels_gt, topk=(1, min(numClasses,5)))
            train_acc.addScalar(acc1, preds.shape[0])
            train_acc_5.addScalar(acc5, preds.shape[0])
            
            loss_train.add(loss)

            if opt.lr_policy !="None":
                scheduler.step()

        # print
        if get_rank() == 0:
            if (iteration + 1) % opt.valInterval == 0 or iteration==0 or opt.testFlag: # To see training progress, we also conduct validation when 'iteration == 0' 
                #validation
                # iCntr=torch.tensor(0.0).to(device)
                # valCorrect=torch.tensor(0.0).to(device)
                vggFontModel.eval()
                print('Inside val',iteration)

                for vCntr, (image_input_tensors, labels_gt) in enumerate(valid_loader):
                    # print('vCntr--',vCntr)
                    if opt.debugFlag and vCntr >2:
                        break
                    
                    
                    with torch.no_grad():
                        image_input_tensors = image_input_tensors.to(device)
                        labels_gt = labels_gt.view(-1).to(device)

                        preds = vggFontModel(image_input_tensors)
                        loss = criterion(preds, labels_gt)
                        loss_val.add(loss)
                        
                        # _, preds_max = preds.max(dim=1)
                        # valCorrect += (preds_max == labels_gt).sum()
                        # iCntr+=preds_max.shape[0]
                        
                        acc1, acc5, acc10 = getNumCorrect(preds, labels_gt, topk=(1, min(numClasses,5), min(numClasses,10)))
                        val_acc.addScalar(acc1, preds.shape[0])
                        val_acc_5.addScalar(acc5, preds.shape[0])
                        val_acc_10.addScalar(acc10, preds.shape[0])
                                                
                vggFontModel.train()    
                elapsed_time = time.time() - start_time

                #DO HERE
                with open(os.path.join(opt.exp_dir,opt.exp_name,'log_train.txt'), 'a') as log:
                    # print('COUNT-------',val_acc_5.n_count)
                    # training loss and validation loss
                    if opt.testFlag:
                        loss_log = f'[{iteration+1}/{opt.num_iter}]  \
                        Val loss: {loss_val.val():0.5f}, \
                        Val Top-1 Acc: {val_acc.val()*100:0.5f}, Val Top-5 Acc: {val_acc_5.val()*100:0.5f}, \
                        Val Top-10 Acc: {val_acc_10.val()*100:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                    else:
                        loss_log = f'[{iteration+1}/{opt.num_iter}]  \
                        Train loss: {loss_train.val():0.5f}, Val loss: {loss_val.val():0.5f}, \
                        Train Top-1 Acc: {train_acc.val()*100:0.5f}, Train Top-5 Acc: {train_acc_5.val()*100:0.5f}, \
                        Val Top-1 Acc: {val_acc.val()*100:0.5f}, Val Top-5 Acc: {val_acc_5.val()*100:0.5f}, \
                        Val Top-10 Acc: {val_acc_10.val()*100:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                    
                    
                    #plotting
                    if not opt.testFlag:
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-Loss'), loss_train.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Val-Loss'), loss_val.val().item())
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-Top-1-Acc'), train_acc.val()*100)
                        lib.plot.plot(os.path.join(opt.plotDir,'Train-Top-5-Acc'), train_acc_5.val()*100)
                    lib.plot.plot(os.path.join(opt.plotDir,'Val-Top-1-Acc'), val_acc.val()*100)
                    lib.plot.plot(os.path.join(opt.plotDir,'Val-Top-5-Acc'), val_acc_5.val()*100)
                    lib.plot.plot(os.path.join(opt.plotDir,'Val-Top-10-Acc'), val_acc_10.val()*100)

                    print(loss_log)
                    log.write(loss_log+"\n")

                    loss_train.reset()
                    loss_val.reset()
                    train_acc.reset()
                    val_acc.reset()
                    train_acc_5.reset()
                    val_acc_5.reset()
                    val_acc_10.reset()
                    # trainCorrect=0
                    # tCntr=0
                    
                lib.plot.flush()

            # save model per 30000 iter.
            if (iteration) % 15000 == 0:
                torch.save({
                'vggFontModel':vggFontModel_module.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict()}, 
                os.path.join(opt.exp_dir,opt.exp_name,'iter_'+str(iteration+1)+'_vggFont.pth'))
                    

            lib.plot.tick()
        
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

def getNumCorrect(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='/checkpoint/pkrishnan/experiments/scribe/Exp17/', help='Where to store logs and models')
    parser.add_argument('--exp_name', default='debug', help='Where to store logs and models')
    parser.add_argument('--train_img_dir', default='/checkpoint/pkrishnan/datasets/synth-datasets/scribe/synthtext-all-font-train/imgs/', help='path to training dataset')
    parser.add_argument('--val_img_dir', default='/checkpoint/pkrishnan/datasets/synth-datasets/scribe/synthtext-all-font-train/imgs/', help='path to training dataset')
    parser.add_argument('--train_ann_file', default='/checkpoint/pkrishnan/datasets/synth-datasets/scribe/synthtext-all-font-train/ann-train.txt', help='path to validation dataset')
    parser.add_argument('--val_ann_file', default='/checkpoint/pkrishnan/datasets/synth-datasets/scribe/synthtext-all-font-train/ann-val.txt', help='path to validation dataset')
    parser.add_argument('--numClasses', type=int, default=-1, help='-1: take all classes from training set')

    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=900000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval between each validation')
    parser.add_argument('--saved_font_model', default='', help="path to model to continue training")
    parser.add_argument('--optim', default='sgd', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--lr_policy', default='None', help='None|step')
    parser.add_argument('--step_size', type=int, default=25000, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='how much to decay learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--preTrained', action='store_true', help='for testing')
    
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=256, help='the width of the input image')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    
    parser.add_argument('--debugFlag', action='store_true', help='for debugging')
    parser.add_argument('--modelFolderFlag', action='store_true', help='load latest files from saved model folder')
    parser.add_argument('--testFlag', action='store_true', help='for testing')

    parser.add_argument("--local_rank", type=int, default=0)
    
    

    opt = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.distributed = n_gpu > 1

    if opt.distributed:
        print("Running distributed setting: Num gpus::", n_gpu)
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    


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

