#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__


# In[2]:


from tensorflow.keras.models import Sequential
import random
from scipy import stats
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed


from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import nltk
import matplotlib.pyplot as plt


# In[3]:


ARABIC_LETTERS = [
  u'ج', u'ح', u'خ', u'ه',u'ع', 
    u'غ', u'ف', u'ق', u'ث', u'ص', u'ض', u'ط',
    u'ك', u'م', u'ن', u'ت', u'ا', u'ل', u'ب', 
    u'ي', u'س', u'ش', u'د', u'ظ', u'ز', u'و', 
    u'ة', u'آ', u'أ', u'إ', u'ئ', u'ى', u'ر', 
    u'ؤ', u'ء', u'ذ', u' '
]

ARABIC_HARKATS = [
  u'ً', u'ٌ', u'ٍ',
  u'َ', u'ُ', u'ِ', u'ّ', u'ْ', u' ', u'ٰ', u'-',
  u'ٖ', u'َّ', u'ُّ', u'ِّ'
]

dictForDoubleHarkats = {u'َّ': 'a', u'ُّ': 'b', u'ِّ': 'c', u'ّٰ': 'd', u'ّٖ': 'e'}

def find_space(s):
    c = 0
    lastX = -1
    for x in range(len(s)-1,0,-1):
        if(s[x] == u' '):
            c = c + 1
            lastX = x
            if(c == 2):
                return x
    return lastX
    
#function that converts a list to string
def listToString(s): 
    str1 = ""     
    for ele in s:  
        str1 += ele      
    return str1  

def removeDiacritics(inSeq):
    lst = list(inSeq)
    L = len(lst)
    for x in range(L):
        if(lst[x] not in ARABIC_LETTERS):
            lst[x] = ''
            
    s = listToString(lst)
    return s


# In[4]:


import unicodedata as ud

file = open("D:\\FAST stuff\\Semester 8\\FYP-II\\train.txt", encoding="utf8")
text = file.read()
file.close()

text = ''.join(c for c in text if not ud.category(c).startswith('P'))
tkns = nltk.word_tokenize(text)
cleansedTkns = []

for x in range(len(tkns)):
    if(len(tkns[x]) > 1):
        cleansedTkns.append(tkns[x])

        
file = open("D:\\FAST stuff\\Semester 8\\FYP-II\\القواعد لابن رجب.txt", encoding="utf8")
text = file.read()
file.close()

text = ''.join(c for c in text if not ud.category(c).startswith('P'))
tkns = nltk.word_tokenize(text)

for x in range(len(tkns)):
    if(len(tkns[x]) > 1):
        cleansedTkns.append(tkns[x])
        
file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\val.txt", encoding="utf8")
text = file.read()
file.close()

text = ''.join(c for c in text if not ud.category(c).startswith('P'))
tkns = nltk.word_tokenize(text)

for x in range(len(tkns)):
    if(len(tkns[x]) > 1):
        cleansedTkns.append(tkns[x])

        
file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\test.txt", encoding="utf8")
text = file.read()
file.close()

text = ''.join(c for c in text if not ud.category(c).startswith('P'))
tkns = nltk.word_tokenize(text)

for x in range(len(tkns)):
    if(len(tkns[x]) > 1):
        cleansedTkns.append(tkns[x])        

tknsProcessed = list(set(cleansedTkns))    


# In[5]:


tknsProcessed.sort()
types = {}

for u in range(len(tknsProcessed)):
    w = removeDiacritics(tknsProcessed[u])
    if(w not in types):
        types[w] = []
    types[w].append(tknsProcessed[u])


# In[6]:


#you need to give the path of the train data file here. 
file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\القواعد لابن رجب.txt", encoding="utf8")
#file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\test.txt", encoding="utf8")
#file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\train2.txt", encoding="utf8")
f = file.read()
file.close()
#do print(f) just to see if the datafile is read correctly

l = len(f)
f = list(f)
for x in range(l):
    if(f[x] == '\n'):
        f[x] = ' '
        
f = listToString(f)
        
def undiacritizeText(fileText):
    l = len(fileText)
    l1 = len(ARABIC_LETTERS)
    l2 = len(ARABIC_HARKATS)
    check = 0
    check2 = 0
    check3 = 0 #to check if the current character was present in either of the letters or diacritics lists
    for i in range(0,l):
        for k in range(0,l1):
            if(fileText[i] == ARABIC_LETTERS[k]):
                check3 = 1
                #print(fileText[i] + '\t' + ARABIC_LETTERS[k])
                check = check + 1
                #print(str(check) + " At 1st if")
                check2 = 0
                global data1
                data1.append(ARABIC_LETTERS[k])
                break
        for s in range(0,l2):
            if(fileText[i] == ARABIC_HARKATS[s]):
                check3 = 1
                global data2
                #print(str(check) + " At 2nd if")
                #if(fileText[i] == ARABIC_HARKATS[9] and check >= 2):
                    #data2.append(ARABIC_HARKATS[10])
                    #data2.append(ARABIC_HARKATS[10])
                if(fileText[i] == ARABIC_HARKATS[8] and check >= 2):
                    data2.append(ARABIC_HARKATS[10])
                    data2.append(ARABIC_HARKATS[10])
                    #print(len(data1))
                    #print(len(data2))
                check2 = check2 + 1
                #if(check == 2 and (fileText[i] != ARABIC_HARKATS[9] and fileText[i] != ARABIC_HARKATS[8])):
                if(check == 2 and (fileText[i] != ARABIC_HARKATS[8])):    
                    #print(fileText[i])
                    data2.append(fileText[i])
                    #print(len(data1))
                    #print(len(data2))
                if(check2 < 2):
                    if(check != 2):
                        if(fileText[i] == ' ' and check == 1):#!= 3
                            data2.append(ARABIC_HARKATS[10])
                        if(fileText[i] != ' '):
                            data2.append(fileText[i])
                        #else:
                            #data2.append(fileText[i])
                elif(check2 == 2):
                    pos = len(data2)-1
                    if(fileText[i-1:i+1] in dictForDoubleHarkats):
                        data2[pos] = dictForDoubleHarkats[fileText[i-1:i+1]]
                
                check = 0
                #print(len(data1))
                #print(len(data2))
                break
        if(check >= 2 and check3 == 1):
            check3 = 0
            #print(str(check) + " At last if")
            data2.append(ARABIC_HARKATS[10])
            #print(len(data1))
            #print(len(data2))
        #print("end")
    if(check == 1):
        data2.append(ARABIC_HARKATS[10])
    

#seq is the sequence of input we talked about
seq = 4 #worked good
#seq = 5
#dataX would be holding all the numeric values of the arabic LETTERS
#dataY would be holding all the numeric values of the arabic DIACRITICS
#data1 would be holding all the arabic LETTERS
#data2 would be holding all the arabic DIACRITICS
#data1 and data2 are used while undiacritizing the file text

data1 = []
data2 = []
undiacritizeText(f)

file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\UndiacritizedText.txt", "w", encoding="utf8")
d1 = listToString(data1)
#print(d1)
file.write(d1)
file.close()

file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\Diacritics.txt", "w", encoding="utf8")
d2 = listToString(data2)
file.write(d2)
file.close()

file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\LengthOfCharacters.txt", "w")
l = str(len(data1))
file.write(l)
file.close()


# In[6]:


dataX = []
dataY = []
#seq = 20
#seq = 4
seq = 5
#seq = 3
file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\LengthOfCharacters.txt")
nChars = int(file.read())
file.close()

file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\UndiacritizedText.txt", encoding="utf8")
data1 = file.read()
file.close()

file = open("D:\\FAST stuff\\Semester 7\\FYP-I\\Diacritics.txt", encoding="utf8")
data2 = file.read()
file.close()

#nChars = len(data1)


#chars = sorted(list(set(listToString(data1))))
#chars1 = sorted(list(set(listToString(data2))))
chars = sorted(list(set(data1)))
chars1 = sorted(list(set(data2)))
#chars = sorted(ARABIC_LETTERS)
#chars1 = sorted(ARABIC_HARKATS)
n_chars = len(chars)
n_chars1 = len(chars1)
#chars1.append('-')

#make dictionary using the list of characters !!!!


#print(data1)
#print(data2)

#for x in range(len(data1)):
    #print(data1[x] + '\t' + data2[x])

#2 dictionaries are initialized, one holds character to number values of arabic LETTERS (which is char_to_int) and 
#the other holds character to number values of arabic DIACRITICS
char_to_int = dict((c,i) for i,c in enumerate(chars))
char_to_int1 = dict((c,i) for i,c in enumerate(chars1))


print(char_to_int)
print()
print(char_to_int1)


print()

file = open("D:\\FAST stuff\\Semester 8\\FYP-II\\TrainingSamples.txt", "a", encoding="utf8")
file1 = open("D:\\FAST stuff\\Semester 8\\FYP-II\\TrainingTargets.txt", "a", encoding="utf8")

x = 0
while(x < nChars - seq):
    seq_in = listToString(data1[x:x + seq])
    file.write(seq_in)
    file.write('\n')
    dataX.append([char_to_int[char] for char in seq_in])
    
    seq_out = listToString(data2[x:x + seq])
    file1.write(seq_out)
    file1.write('\n')
    dataY.append([char_to_int1[char1] for char1 in seq_out])
    #x = x + find_space(data1[x:x + seq]) + 1
    x = x + 1 #testing this, remove this if not working
    
file.close()
file1.close()
    
n_patterns = len(dataX)
#print(dataX[n_patterns-1])
del dataX[n_patterns-1]
del dataY[n_patterns-1]
n_patterns = n_patterns-1
#print()
#print(dataY)
#reshaping and normalizing the input data before feeding it to our model
X = np.reshape(dataX, (n_patterns, seq, 1))
#X = X/float(n_chars)
X = X/float(1)
#print(X)
#X = float(X)

n_patterns = len(dataY)
y = np.reshape(dataY, (n_patterns, seq))
#y = y/float(n_chars1)
y = y/float(1) #try this statement instead of above one

print(len(X))
print(len(y))


# In[8]:


#model = Sequential()
#model.add(Bidirectional(LSTM(200, activation='relu', input_shape=(X.shape[1], X.shape[2]))))
#model.add(RepeatVector(y.shape[1]))
#model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True)))
#model.add(TimeDistributed(Dense(100, activation='relu')))
#model.add(TimeDistributed(Dense(1)))
#model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
# fit network
#model.fit(X, y , epochs=10, verbose=1) #, batch_size=16 worked fine


# In[7]:


model = load_model('modelBLSTMEncodeDecoderSeq5.h5')





