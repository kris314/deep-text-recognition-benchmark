
annFile='/private/home/pkrishnan/datasets/IAM/ascii/words_textsynth.txt'
outFile = '/private/home/pkrishnan/datasets/IAM/ascii/words_iam.txt'

fout = open(outFile,'w')

with open(annFile,'r') as fid:
    lines = fid.readlines()

    for line in lines:
        print(line)
        tokens = line[:-1].split(' ')
        fout.write('%s.png %s\n'%(tokens[0],tokens[-1]))

fout.close()
