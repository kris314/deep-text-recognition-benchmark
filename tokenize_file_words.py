import nltk
import random
nltk.download('stopwords')

from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))


fName = '/private/home/pkrishnan/codes/st-scribe/SynthText/data/newsgroup/newsgroup.txt'
fOutName = '/private/home/pkrishnan/codes/st-scribe/SynthText/data/newsgroup/newsgroup_words_filtered.txt'
minLen = 2
maxLen = 23
totalWords=1000000

fOut = open(fOutName,'w')
# print(en_stops)
cntr=0
with open(fName, 'r') as fid:
    for line in fid.readlines():
        print('%d/%d'%(cntr,totalWords))
        for word in line[:-1].split(' '):

            if not(word.isalnum()) or len(word)<minLen or len(word)>maxLen:
                continue
            if word in en_stops:
                if random.random()>0.50:
                    continue
            # print(word.lower())
            fOut.write('%s\n'%word.lower())
            cntr+=1
        if cntr>totalWords:
            break
fOut.close()






