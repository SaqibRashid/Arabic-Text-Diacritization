from tensorflow import keras
from scipy import stats
import numpy as np
import unicodedata as ud
import nltk
# nltk.download('punkt')

ARABIC_LETTERS = [
    u'ج', u'ح', u'خ', u'ه', u'ع',
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

seq = 5

chars = sorted(ARABIC_LETTERS)
chars1 = sorted(ARABIC_HARKATS)
n_chars = len(chars)
n_chars1 = len(chars1)

int_to_char = dict((i, c) for i, c in enumerate(chars))
int_to_char1 = dict((i, c) for i, c in enumerate(chars1))
char_to_int = dict((c, i) for i, c in enumerate(chars))
char_to_int1 = dict((c, i) for i, c in enumerate(chars1))


def find_space(s):
    c = 0
    lastX = -1
    for x in range(len(s) - 1, 0, -1):
        if s[x] == u' ':
            c = c + 1
            lastX = x
            if c == 2:
                return x
    return lastX


def list_to_string(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1


def remove_diacritics(input_seq):
    lst = list(input_seq)
    l = len(lst)
    for x in range(l):
        if lst[x] not in ARABIC_LETTERS:
            lst[x] = ''

    s = list_to_string(lst)
    return s


def find_mode(lst, numOfDiacritics):
    lst1 = list(filter(lambda a: a != -1, lst))
    mode = stats.mode(lst1)
    mode = mode[0][0]
    if (mode >= numOfDiacritics):
        mode = numOfDiacritics - 1
    return mode


def postProcess(seqSize, inputSize, inputSeq, model1, n_chars):  # n_chars = n_chars1 from main and model1 is model from main
    dX = []
    NumberOfPrediction = inputSize - seqSize + 1  # +1 because if seqsize is 20 and inputsize is 22, we need 3 results
    print('Total samples: ', NumberOfPrediction)

    if NumberOfPrediction == 1:
        seq_in = list_to_string(inputSeq)
        dX.append([char_to_int[char] for char in seq_in])
        xt = np.reshape(dX[0], (1, len(dX[0]), 1))
        xt = xt / float(1)
        prediction = model1.predict(xt)
        l = len(prediction[0])
        h = [0] * l
        prediction = np.rint(prediction)
        for x1 in range(0, l):
            h[x1] = int(prediction[0][x1])
            if h[x1] < 0:
                h[x1] = 0
        prediction = h
        return prediction

    Arr = [-1] * NumberOfPrediction
    for x in range(0, NumberOfPrediction):
        if x % 1000 == 0:
            print('sample: ', x)

        Arr[x] = [-1] * inputSize
        seq_in = list_to_string(inputSeq[x:x + seqSize])
        dX.append([char_to_int[char] for char in seq_in])
        xt = np.reshape(dX[x], (1, len(dX[x]), 1))
        xt = xt / float(1)
        prediction = model1.predict(xt)

        l = len(prediction[0])
        h = [0] * l
        prediction = np.rint(prediction)
        prediction = prediction.astype(int)
        prediction = prediction[0]
        z = 0
        for y in range(x, seqSize + x):
            if prediction[z] < 0:
                prediction[z] = 0
            Arr[x][y] = prediction[z]
            z = z + 1

    listToReturn = [0] * inputSize
    for z in range(0, inputSize):
        listToPass = [0] * NumberOfPrediction
        for x in range(0, NumberOfPrediction):
            listToPass[x] = Arr[x][z]
        listToReturn[z] = find_mode(listToPass, n_chars + 1)

    return listToReturn


file = open("train.txt", encoding="utf8")
text = file.read()
file.close()

text = ''.join(c for c in text if not ud.category(c).startswith('P'))
tkns = nltk.word_tokenize(text)
cleansedTkns = []

for x in range(len(tkns)):
    if len(tkns[x]) > 1:
        cleansedTkns.append(tkns[x])

file = open("القواعد لابن رجب.txt", encoding="utf8")
text = file.read()
file.close()

text = ''.join(c for c in text if not ud.category(c).startswith('P'))
tkns = nltk.word_tokenize(text)

for x in range(len(tkns)):
    if len(tkns[x]) > 1:
        cleansedTkns.append(tkns[x])

file = open("val.txt", encoding="utf8")
text = file.read()
file.close()

text = ''.join(c for c in text if not ud.category(c).startswith('P'))
tkns = nltk.word_tokenize(text)

for x in range(len(tkns)):
    if len(tkns[x]) > 1:
        cleansedTkns.append(tkns[x])

file = open("test.txt", encoding="utf8")
text = file.read()
file.close()

text = ''.join(c for c in text if not ud.category(c).startswith('P'))
tkns = nltk.word_tokenize(text)

for x in range(len(tkns)):
    if len(tkns[x]) > 1:
        cleansedTkns.append(tkns[x])

tknsProcessed = list(set(cleansedTkns))

# In[5]:


tknsProcessed.sort()
types = {}

for u in range(len(tknsProcessed)):
    w = remove_diacritics(tknsProcessed[u])
    if w not in types:
        types[w] = []
    types[w].append(tknsProcessed[u])

dictForDoubleHarkatsReverse = {'a': u'َّ', 'b': u'ُّ', 'c': u'ِّ', 'd': u'ّٰ', 'e': u'ّٖ'}

testInput = open("input.txt", encoding="utf8").read()
# print(testInput)
testInput = testInput.replace('\xa0', ' ')
testInput = remove_diacritics(testInput)  # even if input is semi diacritized it will work
lengthOfInput = len(testInput)
originalInput = nltk.word_tokenize(testInput)

if lengthOfInput < seq:
    for x in range(lengthOfInput, seq):
        testInput = testInput + ' '

# working for any size input sequences now

lengthOfInput = len(testInput)

model = keras.models.load_model('model_encoder_2.h5', compile=False)
diacritics = postProcess(seq, lengthOfInput, testInput, model, n_chars1 - 1)
l = len(diacritics)
# print("Got Predictions")
nc = sorted(list(set(testInput)))
dX = [0] * lengthOfInput
for i in range(0, lengthOfInput):
    dX[i] = char_to_int[testInput[i]]

lst = [''] * l
lst1 = [''] * l
# print("Starting with decoding")
for x in range(0, l):
    if l <= seq:
        lst[x] = int_to_char1[diacritics[x]]
    else:
        lst[x] = int_to_char1[diacritics[x][0]]
    if lst[x] == '-':
        lst[x] = ''
    lst1[x] = int_to_char[dX[x]]

check_skip = 0
textResults = ''
# print("Merging with user input")
for x in range(0, l):
    c = lst[x]
    if lst1[x] == ' ' or (c == ' ' and lst1[x] == ' ') or c == ' ':
        c = ''

    if lst1[x] == 'أ' or lst1[x] == 'إ' or lst1[x] == 'ا':
        if c != ' ':
            c = ''

    if check_skip == 1:
        c = ''
        check_skip = 0

    if x < l - 1:
        if lst1[x] == 'ا' and lst1[x + 1] == 'ل':
            check_skip = 1
        if (lst1[x] == 'ي' or lst1[x] == 'ى') and lst1[x + 1] == ' ':
            c = ''

    if c in dictForDoubleHarkatsReverse:
        c = dictForDoubleHarkatsReverse[c]
    textResults = textResults + lst1[x] + c

tokenizedTextResults = nltk.word_tokenize(textResults)
afterProcessing = ''
for x in range(len(originalInput)):
    w = tokenizedTextResults[x]
    if originalInput[x] in types:
        candidateWords = types[originalInput[x]]
        candidates = {}

        for z in range(len(candidateWords)):
            d = nltk.edit_distance(w, candidateWords[z])
            if candidateWords[z] not in candidates:
                candidates[candidateWords[z]] = d

        for y in range(len(candidates)):
            key_min = min(candidates.keys(), key=(lambda k: candidates[k]))
            ud = remove_diacritics(key_min)
            if ud == originalInput[x]:
                break
            del candidates[key_min]

        if x < len(originalInput) - 1:
            afterProcessing = afterProcessing + key_min + ' '
        else:
            afterProcessing = afterProcessing + key_min
    else:
        if x < len(originalInput) - 1:
            afterProcessing = afterProcessing + w + ' '
        else:
            afterProcessing = afterProcessing + w

file = open("prediction.txt", "w", encoding="utf8")
file.write(afterProcessing)
file.close()
