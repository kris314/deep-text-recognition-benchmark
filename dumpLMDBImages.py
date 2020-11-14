import lmdb
import sys
import re
from PIL import Image
import os
import pdb
import six
import numpy as np

if __name__ == "__main__":

    # root = '/checkpoint/pkrishnan/datasets/icdar15_test/'
    # text_file = '/checkpoint/pkrishnan/datasets/synth-datasets/scribe/newsgroup_words_vqa.txt'
    # text_file = '/checkpoint/pkrishnan/datasets/synth-datasets/scribe/newsgroup_words_test.txt'

    # outImgFolder='/checkpoint/pkrishnan/datasets/icdar15_test_images/'
    # outPairFile='/checkpoint/pkrishnan/datasets/icdar15_test_pairfile.txt'

    root=sys.argv[1]
    text_file=sys.argv[2]
    outImgFolder=sys.argv[3]
    outPairFile=sys.argv[4]
    dumpImageFlag = 1

    if not(os.path.exists(outImgFolder)):
        os.makedirs(outImgFolder)

    fout = open(outPairFile,'w')

    env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

    if not env:
        print('cannot create lmdb from %s' % (root))
        sys.exit(0)

    wordIdx={}
    cntr=0
    # pdb.set_trace()
    with open(text_file,'r') as fid:
        lines = fid.readlines()
        unqWords = []

        for currWord in lines:
            print(len(lines),cntr)
            if not(currWord[:-1] in unqWords):
                unqWords.append(currWord[:-1].lower())
                
                if not(len(currWord[:-1]) in wordIdx):
                    wordIdx[len(currWord[:-1])] = []
                    wordIdx[len(currWord[:-1])].append(len(unqWords)-1)
                else:
                    wordIdx[len(currWord[:-1])].append(len(unqWords)-1)
            cntr+=1
    # pdb.set_trace()
    with env.begin(write=False) as txn:
        
        nSamples = int(txn.get('num-samples'.encode()))
        print('nSamples:::::::::::::',nSamples)

        # self.filtered_index_list = []
        for index in range(nSamples):
            print(nSamples,index)
            index += 1  # lmdb starts with 1
            
            #read image and filter
            label_string = 'label-%09d' % index
            label1_key = 'label-%09d'.encode() % index
            label1 = txn.get(label1_key).decode('utf-8')
            
            if dumpImageFlag:
                img1_key = 'image-%09d'.encode() % index
                img1buf = txn.get(img1_key)

                buf1 = six.BytesIO()
                buf1.write(img1buf)
                buf1.seek(0)

            try:
                numChars = len(label1)
                if numChars<3 or numChars>max(wordIdx.keys()):
                    continue
                len_array = np.arange(max(3,numChars-3),min(max(wordIdx.keys()),numChars+3))
                # if len_array.shape[0] == 0:
                #     continue
                defaultProbs = 0.2/(len_array.shape[0]-1)
                prob_array = np.ones(len_array.shape[0])*defaultProbs
                idx = np.where(len_array==numChars)[0][0]
                prob_array[idx] = 0.8

                sample_len = np.random.choice(len_array,1,p=prob_array)
                idx = np.random.choice(len(wordIdx[sample_len[0]]),1)[0]
                sample_text =unqWords[wordIdx[sample_len[0]][idx]]

                #set case
                if label1.isupper():
                    if np.random.random()>0.1:
                        sample_text = sample_text.upper()
                elif label1[0].isupper():
                    sample_text = sample_text[0].upper()+sample_text[1:]
                else:
                    if np.random.random()>0.9:
                        sample_text = sample_text.upper()
                if dumpImageFlag:
                    img1 = Image.open(buf1).convert('RGB')  # for color image
                    img1.save(os.path.join(outImgFolder,label_string+'.png'))
                    
                fout.write('%s %s %s\n'%(label_string, label1 , sample_text))
            except:
                print(f'Corrupted image for {index}')
                continue

    fout.close()
        

        