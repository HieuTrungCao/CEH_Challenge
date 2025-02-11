from PyPDF2 import PdfReader
import pandas as pd

import pandas as pd
import re
# from .util import preprocess
import re, string, unicodedata
import nltk
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "url", sample)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    # words = remove_non_ascii(words)
    words = to_lowercase(words)
    # words = remove_punctuation(words)
    # words = replace_numbers(words)
    # words = remove_stopwords(words)
    return words

def preprocess(sample):
    sample = remove_URL(sample)
    # sample = replace_contractions(sample)
    # Tokenize
    words = nltk.word_tokenize(sample)

    # Normalize
    return " ".join(normalize(words))

reader = PdfReader("data/GereralAI/Sybex_CEH_v10_Certified_Ethical.pdf")

question = []
answers = []

for page in  reader.pages[40: ]:
    sentences = preprocess(page.extract_text()).replace(".\n ", "").replace("\n", " ").split(".")
    for sentence in sentences[1 : -1]:
        sentence = sentence.strip()
        if (
            len(sentence) > 100 
            and isinstance(sentence, str) 
            and not "   " in sentence
            and not "* * *" in sentence
            and len(sentence) < 500):
            answers.append(sentence) 
            q = " ".join(sentence.split(" ")[: int(len(sentence.split(" ")) / 2)])
            question.append(q)

data = {
    "sentence": question,
    "answer": answers
}

data = pd.DataFrame(data)
data.to_csv("./data/GereralAI/train/my_data.csv", index=False)