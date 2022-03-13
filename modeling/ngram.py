#Module for creating language corpuses

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import string
from collections import defaultdict
from string import punctuation
# nltk.download('stopwords')

N_GRAM = 3

def remove_punctuation(text):
    if(type(text)==float):
        return text
    ans=""  
    for i in text:     
        if i not in string.punctuation:
            ans+=i 
    return ans

def generate_N_grams(text, language, ngram=1):
    words=[word for word in text.split(" ") if word not in set(stopwords.words(language))]  
    temp=zip(*[words[i:] for i in range(0,ngram)])
    ans=[' '.join(ngram) for ngram in temp]
    return ans

def genereate_corpuses(filename, n_gram) -> defaultdict(list):
    """
    Parameters
    ----------
    filename : str
        csv file, located in the same direction, with two columns:
            -- "Text": Text in some language
            -- "Language": Language of a given text
    n_gram : int
        Length on n-grams to generate

    Returns
    -------
    defaultdict(list)
        default dict with languages as keys and lists of n-grams as values
    """
    df = pd.read_csv(filename)
    df.Language = df.Language.apply(lambda x: x.lower())

    #extracting the languages, for which stopwords are available
    deleting_indexes = df[df.Language.map(lambda x: x not in stopwords.fileids())].index
    df.drop(deleting_indexes, inplace=True)
    
    #removing punktuation
    df['text']= df['Text'].apply(lambda x:remove_punctuation(x))

    corpuses = defaultdict(list)
    for index, row in df.iterrows():
        corpuses[row.Language] += generate_N_grams(row.Text, row.Language, N_GRAM)
    return corpuses