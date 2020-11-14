import cv2
import os
import glob
import numpy as np
import shutil

import pdb

numImg=60
pattern='*_pred_val_c2_s1_*'

#ICDAR15
# infolder_srnet = '/checkpoint/pkrishnan/experiments/scribe/Exp18/userstudy/srnet/20201020150557_icdar15/'
infolder_ours = '/checkpoint/pkrishnan/experiments/scribe/Exp18/userstudy/iam_all_58_150001_run3/testImages/150001/'

# #TextVQA
# infolder_srnet = '/checkpoint/pkrishnan/experiments/scribe/Exp18/userstudy/srnet/20201020150557_textvqa/'
# infolder_ours = '/checkpoint/pkrishnan/experiments/scribe/Exp18/userstudy/textvqa_all_13_690001/testImages/690001/'

# ICDAR15
outFolder_real='/checkpoint/pkrishnan/experiments/scribe/Exp18/userstudy/hw-subset/iam_real/'
outFolder_ours='/checkpoint/pkrishnan/experiments/scribe/Exp18/userstudy/hw-subset/iam_ours/'
# outFolder_srnet='/checkpoint/pkrishnan/experiments/scribe/Exp18/userstudy/subset-112/icdar15_srnet/'

# #textvqa
# outFolder_ours='/checkpoint/pkrishnan/experiments/scribe/Exp18/userstudy/subset/textvqa_all_13_690001/'
# outFolder_srnet='/checkpoint/pkrishnan/experiments/scribe/Exp18/userstudy/subset/textvqa_srnet/'

if not(os.path.exists(outFolder_ours)):
    os.makedirs(outFolder_ours)

# if not(os.path.exists(outFolder_srnet)):
#     os.makedirs(outFolder_srnet)

if not(os.path.exists(outFolder_real)):
    os.makedirs(outFolder_real)


fnames = glob.glob(infolder_ours+pattern+'.png')

idx = np.random.choice(range(len(fnames)), numImg, replace=False)

for i in idx:
    
    infile_ours = os.path.basename(fnames[i])
    text = infile_ours.split('_')[-3]
    temp = infile_ours.split('-')
    labelFile = temp[0]+'-'+temp[1]+'.png'
    targetfile = temp[0]+'-'+temp[1]+'_'+text+'_.png'
    
    infile_real = glob.glob(infolder_ours+temp[0]+'-'+temp[1]+'*_gt_val_c1_s1_*'+'.png')
    
    shutil.copyfile(os.path.join(infolder_ours,infile_ours), os.path.join(outFolder_ours,targetfile))
    # shutil.copyfile(os.path.join(infolder_srnet,labelFile), os.path.join(outFolder_srnet,targetfile))
    shutil.copyfile(infile_real[0], os.path.join(outFolder_real,targetfile))




