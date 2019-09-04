import os
import random
import glob

loc = "/home/aih04/dataset/RealData"
os.chdir(loc)
languages = {'TAM': '0','GUJ': '1','MAR': '2','HIN': '3', 'TEL': '4'}
trainFile = '/home/aih04/LID3/trainInput.txt'
testFile = '/home/aih04/LID3/testInput.txt'
f1 = open(trainFile,'w+')
f2 = open(testFile,'w+')

sampling = 16000
n = 1
trainFrac = 0.9
files = []

for file in glob.glob('*.wav'):
    files.append(file)
    
random.shuffle(files)
index = int(len(files)*trainFrac)
trainFiles = files[0:index]
testFiles = files[index + 1:-1]

for file in trainFiles:
    lang = file[5:8]
    langNo = languages[lang]
    os.rename(file, loc + '/' + str(n) + '.wav')
    n += 1
    line = str(int(n)) + ' ' + str((langNo))
    f1.write(line + '\n')
f1.close()

for file in testFiles:
    lang = file[5:8]
    langNo = languages[lang]
    os.rename(file, loc + '/' + str(n) + '.wav')
    n += 1
    line = str(int(n)) + ' ' + str((langNo))
    f2.write(line + '\n')
f2.close()