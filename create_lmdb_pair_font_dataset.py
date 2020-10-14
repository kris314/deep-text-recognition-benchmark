""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        image1Path, image2Path, label1, label2, font1, font2 = datalist[i].strip('\n').split('\t')
        
        image1Path = os.path.join(inputPath, image1Path)
        image2Path = os.path.join(inputPath, image2Path)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not(os.path.exists(image1Path)) or not(os.path.exists(image2Path)):
            print('%s or %s does not exist' % (image1Path, image2Path))
            continue
        with open(image1Path, 'rb') as f:
            image1Bin = f.read()
        with open(image2Path, 'rb') as f:
            image2Bin = f.read()
        if checkValid:
            try:
                if not(checkImageIsValid(image1Bin)) or not(checkImageIsValid(image2Bin)):
                    print('%s or %s is not a valid image' % (image1Path, image2Path))
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        image1Key = 'image1-%09d'.encode() % cnt
        image2Key = 'image2-%09d'.encode() % cnt
        label1Key = 'label1-%09d'.encode() % cnt
        label2Key = 'label2-%09d'.encode() % cnt
        font1Key = 'font1-%09d'.encode() % cnt
        font2Key = 'font2-%09d'.encode() % cnt

        cache[image1Key] = image1Bin
        cache[image2Key] = image2Bin
        cache[label1Key] = label1.encode()
        cache[label2Key] = label2.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset)
