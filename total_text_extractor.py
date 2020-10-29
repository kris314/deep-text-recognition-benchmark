import os
import scipy.io as io
import pdb
import cv2

imgDir = '/checkpoint/pkrishnan/datasets/totaltext/Images/Test'
gtDir = '/checkpoint/pkrishnan/datasets/totaltext/Groundtruth/Rectangular/Test'
outImgFolder = '/checkpoint/pkrishnan/datasets/totaltext/Images_Words/Train'
outGTFile = '/checkpoint/pkrishnan/datasets/totaltext/Images_Words/test_ann.txt'

if not(os.path.exists(outImgFolder)):
    os.makedirs(outImgFolder)

fNames = os.listdir(imgDir)

outGTF = open(outGTFile,'w')

for currImgFile in fNames:
    print(currImgFile)

    img_name = currImgFile.split('.')[0]
    currGTFile = 'rect_gt_' + img_name + '.mat'
    mat = io.loadmat(os.path.join(gtDir,currGTFile))
    
    img = cv2.imread(os.path.join(imgDir,currImgFile),cv2.IMREAD_COLOR)
    
    for i in range(len(mat['rectgt'])):
        
        try:
            xmin=mat['rectgt'][i][0].squeeze()
            ymin=mat['rectgt'][i][1].squeeze()
            xmax=mat['rectgt'][i][2].squeeze()
            ymax=mat['rectgt'][i][3].squeeze()
            
            width=mat['rectgt'][i][4].squeeze()
            height=mat['rectgt'][i][5].squeeze()

            text = mat['rectgt'][i][6][0]

            if text=='#':
                continue

            wordImg = img[ymin:ymax,xmin:xmax,:]

            cv2.imwrite(os.path.join(outImgFolder, img_name + '_' + str(i) + '.jpg'), wordImg)
            outGTF.write('%s %s\n'%(img_name + '_' + str(i) + '.jpg', text))
        except:
            print('Could not read file', currImgFile)

outGTF.close()
    