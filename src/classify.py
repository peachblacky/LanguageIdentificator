from pickletools import float8
import sys
sys.path.insert(1, 'C:/Users/Sibdroid/Desktop/6sem/TBIP/LanguageIdentificator')
from src.ngram import n_gram
import numpy as np
from collections import defaultdict
from sklearn.naive_bayes import BernoulliNB


def CFA(corpuses, input, n_gram_length) -> str:
    '''
    Comulative freqency addition classifier

    Parameters
    ----------
    corpuses : defaultdict(list)
        default dict with languages as keys and lists of n-grams as values
    input : str
        text to be analysed
    n_gram_length : int
        length of ngram
    Returns
    -------
    str:   
        presumed language of the text
    '''
    input_ngram = n_gram(input, n_gram_length)
    com_freqs = defaultdict(int)
    for k, v in corpuses.items():
        com_freqs[k] = len(np.intersect1d(v,input_ngram))
        # print('for ' + k + ' cf is ' + str(com_freqs.get(k)))
    return max(com_freqs, key=com_freqs.get)

def MNB(corpuses, input, n_gram_length) -> str:
    '''
    Multinomial Naive Bayes algorithm

    Parameters
    ----------
    corpuses : defaultdict(list)
        default dict with languages as keys and lists of n-grams as values
    input : str
        text to be analysed
    n_gram_length : int
        length of ngram

    Returns
    -------
    str:   
        presumed language of the text
    '''
    input_ngram = n_gram(input, n_gram_length)
    norm_val = len(input_ngram)
    lang_probs = defaultdict(float)
    for k, v in corpuses.items():
        prob = 0
        for ngram in np.intersect1d(v,input_ngram):
            prob += input_ngram.count(ngram) / norm_val
        lang_probs[k] = prob
        print(k + " probability is " + str(prob))
    return max(lang_probs, key=lang_probs.get)

        
        
