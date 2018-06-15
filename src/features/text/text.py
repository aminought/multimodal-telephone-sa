# -*- coding: utf-8 -*-

from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pymorphy2
from nltk.corpus import stopwords
import pickle


def normalize_word(word, morph):
    return morph.parse(word)[0].normal_form


def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)


def remove_stopwords(tokens, stopwords):
    return list(filter(lambda t: t not in stopwords, tokens))


def get_pos(word, morph):
    return morph.parse(word)[0].tag.POS


def lemmatize(text, morph):
    tokens = tokenize_text(text)
    rus_stopwords = [s for s in stopwords.words('russian')
                     if s not in [u'хорошо', u'да', u'нет', u'можно']]
    stops = set(stopwords.words('english')) | set(rus_stopwords)
    clear_tokens = remove_stopwords(tokens, stops)
    poses = [get_pos(t, morph) for t in clear_tokens]
    norm_tokens = [normalize_word(t, morph) for t in clear_tokens]
    return ' '.join(norm_tokens), poses


def preprocess(dataset):
    texts = dataset['train']['text'] + dataset['test']['text']
    lem_texts = []
    poses = []
    morph = pymorphy2.MorphAnalyzer()
    for text in tqdm(texts):
        lem_text, pos = lemmatize(text, morph)
        lem_texts.append(lem_text)
        poses.append(pos)
    return lem_texts, poses


def build_tfidf_features(dataset):
    lem_texts, poses = preprocess(dataset)
    vect = TfidfVectorizer(min_df=0.001)
    features = vect.fit_transform(lem_texts)
    with open('data/processed/text.tfidf.pkl', 'wb') as f:
        pickle.dump(features, f)


def build_bow_features(dataset):
    lem_texts, poses = preprocess(dataset)
    vect = CountVectorizer(ngram_range=(1, 1), min_df=3)
    features = vect.fit_transform(lem_texts)
    with open('data/processed/text.bow.pkl', 'wb') as f:
        pickle.dump(features, f)
