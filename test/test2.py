import sys
sys.path.insert(1, 'somepath/LanguageIdentificator')
from src import classify, ngram
import codecs

N_GRAM_LEN = 1

#Test cases for MNB clssification method


print('Test 1: Guessing english')
f = codecs.open('test/text_english.txt', encoding='utf-8')
data = f.read()
f.close()

language_corpuses = ngram.genereate_corpuses('src/languages.csv', N_GRAM_LEN)
print(classify.MNB(language_corpuses, data, N_GRAM_LEN))

print('Test 2: Guessing german')
f = codecs.open('test/text_deutsch.txt', encoding='utf-8')
data1 = f.read()
f.close()

print(classify.MNB(language_corpuses, data1, N_GRAM_LEN))

print('Test 3: Guessing russian')
f = codecs.open('test/text_russian.txt', encoding='utf-8')
data2 = f.read()
f.close()

print(classify.MNB(language_corpuses, data2, N_GRAM_LEN))

print('Test 4: Guessing arabic')
f = codecs.open('test/text_arabic.txt', encoding='utf-8')
data2 = f.read()
f.close()

print(classify.MNB(language_corpuses, data2, N_GRAM_LEN))