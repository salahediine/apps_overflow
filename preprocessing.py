
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


import re
import spacy
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook().pandas()
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def delete_(s):
    """
    s : la chaine de caracteres à modifier
    """
    soup = BeautifulSoup(s, 'lxml')
    
    l = soup.findAll('code')
    for code in l:
        code.replace_with(' ')
    return str(soup)



nlp = spacy.load("en_core_web_sm")

def text_cleaner(x, nlp, pos):
    
    
    stp_wds = [word for word in stopwords.words('english')]
    x = x.lower()
    x = x.encode("ascii", "ignore").decode()
    x = re.sub('[^\\w\\s#]', '', x) # ponctuation
    x = re.sub(r'http*\S+', '', x) #links
    x = re.sub(r'\w*\d+\w*', '', x) # numbers
    x = re.sub("\'\w+", '', x) #english contraction
    x = re.sub('\s+', ' ', x) #extra space
    
    
    x = nltk.tokenize.word_tokenize(x) # token
    
    x = [word for word in x if word not in stp_wds #stop words
                      and len(word) >2]
    
    x = [WordNetLemmatizer().lemmatize(word) for word in x]
    
    
    return x 










