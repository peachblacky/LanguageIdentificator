import sys
sys.path.insert(1, 'C:/Users/Sibdroid/Desktop/6sem/TBIP/LanguageIdentificator')
from src.ngram import n_gram
import numpy as np
from collections import defaultdict


def CFA(corpuses, input, n_gram_length) -> str:
    '''
    Comulative freqency addition classifier

    Parameters
    ----------
    corpuses : defaultdict(list)
        default dict with languages as keys and lists of n-grams as values
    input : str
        text to be analysed
    
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

def MNB():
    '''
    Multinomial Naive Bayes algorithm
    Parameters
    ----------

    '''