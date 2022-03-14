# Module for creating language corpuses

import pandas as pd
import nltk
from collections import defaultdict
from string import punctuation


def remove_punctuation(text):
    if(type(text) == float):
        return text
    ans = ""
    for i in text:
        if i not in punctuation:
            ans += i
    return ans


def n_gram(text, n_gram_length=1):
    words = remove_punctuation(text).split(" ")
    temp = zip(*[words[i:] for i in range(0, n_gram_length)])
    ans = [' '.join(n_gram) for n_gram in temp]
    return ans


def genereate_corpuses(filepath, n_gram_length) -> defaultdict(list):
    """
    Parameters
    ----------
    filepath : str
        relative path to csv file, located in project, with two columns:
            -- "Text": Text in some language
            -- "Language": Language of a given text
    n_gram_length : int
        Length on n-grams to generate

    Returns
    -------
    defaultdict(list)
        default dict with languages as keys and lists of n-grams as values
    """
    df = pd.read_csv(filepath)
    df.Language = df.Language.apply(lambda x: x.lower())

    # removing punktuation
    df['text'] = df['Text'].apply(lambda x: remove_punctuation(x))

    corpuses = defaultdict(list)
    for index, row in df.iterrows():
        corpuses[row.Language] += n_gram(
            row.Text, n_gram_length)
        
    return corpuses
