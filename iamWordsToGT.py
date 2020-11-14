
annFile='/private/home/pkrishnan/datasets/IAM/ascii/words_textsynth.txt'

trainidx = '/private/home/pkrishnan/datasets/IAM/largeWriterIndependentTextLineRecognitionTask/trainset_textsynth.txt'
validx = '/private/home/pkrishnan/datasets/IAM/largeWriterIndependentTextLineRecognitionTask/valset_textsynth.txt'
testidx = '/private/home/pkrishnan/datasets/IAM/largeWriterIndependentTextLineRecognitionTask/testset.txt'

outTrFile = '/private/home/pkrishnan/datasets/IAM/ascii/iam_trainset_textsynth.txt'
outVaFile = '/private/home/pkrishnan/datasets/IAM/ascii/iam_valset_textsynth.txt'
outTeFile = '/private/home/pkrishnan/datasets/IAM/ascii/iam_testset_textsynth.txt'

#read ids
trainIds=[]
with open(trainidx,'r') as fid:
    lines = fid.readlines()
    for line in lines:
        trainIds.append(line[:-1])

valIds=[]
with open(validx,'r') as fid:
    lines = fid.readlines()
    for line in lines:
        valIds.append(line[:-1])

testIds=[]
with open(testidx,'r') as fid:
    lines = fid.readlines()
    for line in lines:
        testIds.append(line[:-1])

fTrout = open(outTrFile,'w')
fVaout = open(outVaFile,'w')
fTeout = open(outTeFile,'w')

with open(annFile,'r') as fid:
    lines = fid.readlines()

    for line in lines:
        print(line)
        tokens = line[:-1].split(' ')

        iTokens = tokens[0].split('-')
        currId =iTokens[0]+'-'+iTokens[1]+'-'+iTokens[2]

        if currId in trainIds:
            fTrout.write('%s.png %s\n'%(tokens[0],tokens[-1]))
        elif currId in valIds:
            fVaout.write('%s.png %s\n'%(tokens[0],tokens[-1]))
        elif currId in testIds:
            fTeout.write('%s.png %s\n'%(tokens[0],tokens[-1]))    

fTrout.close()
fVaout.close()
fTeout.close()
