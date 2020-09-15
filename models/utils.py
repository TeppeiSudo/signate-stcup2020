import os
import numpy as np
import pandas as pd
import torch
import random
import tqdm
import re
import unicodedata
import nltk
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

# set random seed
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def normalize(text):
    normalized_text = normalize_unicode(text)
    normalized_text = normalize_number(normalized_text)
    normalized_text = lower_text(normalized_text)
    return normalized_text


def lower_text(text):
    return text.lower()


def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text


def lemmatize_term(term, pos=None):
    if pos is None:
        synsets = wordnet.synsets(term)
        if not synsets:
            return term
        pos = synsets[0].pos()
        if pos == wordnet.ADJ_SAT:
            pos = wordnet.ADJ
    return nltk.WordNetLemmatizer().lemmatize(term, pos=pos)


def normalize_number(text):
    """
    pattern = r'\d+'
    replacer = re.compile(pattern)
    result = replacer.sub('0', text)
    """
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text


stemmer = PorterStemmer()
wnl = WordNetLemmatizer()
shortened = {
    '\'m': ' am',
    '\'re': ' are',
    'don\'t': 'do not',
    'doesn\'t': 'does not',
    'didn\'t': 'did not',
    'won\'t': 'will not',
    'wanna': 'want to',
    'gonna': 'going to',
    'gotta': 'got to',
    'hafta': 'have to',
    'needa': 'need to',
    'outta': 'out of',
    'kinda': 'kind of',
    'sorta': 'sort of',
    'lotta': 'lot of',
    'lemme': 'let me',
    'gimme': 'give me',
    'getcha': 'get you',
    'gotcha': 'got you',
    'letcha': 'let you',
    'betcha': 'bet you',
    'shoulda': 'should have',
    'coulda': 'could have',
    'woulda': 'would have',
    'musta': 'must have',
    'mighta': 'might have',
    'dunno': 'do not know',
    }
shortened_re = re.compile('(?:' + '|'.join(map(lambda x: '\\b' + x + '\\b', shortened.keys())) + ')')


def check_except_words(sentence):
    except_words = ["e.g.", "node. ", "Approx. ", "St.", "etc. ", "i.e. ", "Sr. ", "ex. ", "U.S. ", "incl. ", ". (", "I.e. " ]
    for word in except_words:
        if word in sentence:
            return True
    return False


def split_sentences(df):
    sentences = []
    ids = []
    jobs = []
    for n in range(len(df)):
        if ". " in df.description.iloc[n]:
            multi_sentence = []
            if not check_except_words(df.description.iloc[n]):
                multi_sentence.append(list(map(str,df.description.iloc[n].split(". "))))
                for sentence in multi_sentence[0]:
                    sentences.append(sentence)
                    ids.append(df.iloc[n].id)
                    jobs.append(df.iloc[n].jobflag)
    special_dataset = pd.DataFrame({"id": ids, "description": sentences, "jobflag": jobs})
    return special_dataset


def drop_duplicated_sentence(train):
    df = train.copy()
    # df_dup = df[df.duplicated(subset=['description', 'jobflag'])]
    # df_evil = df[df.duplicated(subset='description', keep=False)]

    for n in tqdm(range(len(df))):
        df.description.iloc[n] = normalize(df.description.iloc[n])
        df.description.iloc[n] = shortened_re.sub(lambda x: shortened[x.group(0)], df.description.iloc[n])
        sentence = ""
        for word in df.description.iloc[n].strip(".:;'&%$#=~|!'<>?*+`@/,_][}{").split(" "):
            word = stemmer.stem(word)
            word = wnl.lemmatize(word)
            sentence += word + " "
        df.description.iloc[n] = sentence[:-1]

    df_dup = df[df.duplicated(subset=['description', 'jobflag'])]
    new_df = df.drop_duplicates(subset=['description', 'jobflag'])
    new_index = new_df.index
    # ただの表記揺れだが、取り除いておく
    new_train = train.reindex(index=new_index)

    df_evil = new_df[new_df.duplicated(subset='description', keep=False)]
    evil_index = df_evil.index
    # descriptionが同一だがjobflagが不一致
    df_evil = train.reindex(index=evil_index)
    print(f'there are {len(df_dup)} duplicates and {len(df_evil)} evil-data... ')

    df_clean = df.drop_duplicates(subset=['description', 'jobflag'])
    df_clean = df_clean.drop_duplicates(subset='description', keep=False)
    clean_index = df_clean.index
    # 全部取り除いたやつ
    clean_train = train.reindex(index=clean_index)

    return new_train, df_evil, clean_train


def split_sentences_for_test(df):
    sentences = []
    ids = []
    jobs = []
    for n in range(len(df)):
        if ". " in df.description.iloc[n]:
            multi_sentence = []
            if not check_except_words(df.description.iloc[n]):
                multi_sentence.append(list(map(str,df.description.iloc[n].split(". "))))
                for sentence in multi_sentence[0]:
                    sentences.append(sentence)
                    ids.append(df.iloc[n].id)
    special_dataset = pd.DataFrame({"id": ids, "description": sentences})
    return special_dataset


# Jaccard係数(overlap coefficiant)の計算アルゴリズム
def jaccard_similarity_coefficient(df_a, df_b):
    vectorizer = CountVectorizer()

    vectorizer.fit_transform(df_a.description)
    list_a = vectorizer.vocabulary_.keys()

    vectorizer.fit_transform(df_b.description)
    list_b = vectorizer.vocabulary_.keys()

    # 集合Aと集合Bの積集合(set型)を作成
    set_intersection = set.intersection(set(list_a), set(list_b))
    # 集合Aと集合Bの積集合の要素数を取得
    num_intersection = len(set_intersection)

    # 集合Aと集合Bの和集合(set型)を作成
    set_union = set.union(set(list_a), set(list_b))
    # 集合Aと集合Bの和集合の要素数を取得
    num_union = len(set_union)

    # 積集合の要素数を和集合の要素数で割って
    # Jaccard係数を算出
    try:
        return float(num_intersection) / num_union
    except ZeroDivisionError:
        return 1.0


def caluculate_jaccard(df_paths, original_df):
    df_a = original_df
    j_dict = {}
    for df_path in df_paths:
        df_b = pd.read_csv(df_path).dropna()
        path = df_path.split('-')[-1].split('.')[0]
        j_score = jaccard_similarity_coefficient(df_a, df_b)
        j_dict[path] = j_score
    return j_dict


def get_jaccard_matrix(df_paths):
    j_dict = {}
    for df_a_path in tqdm(df_paths):
        df_a = pd.read_csv(df_a_path).dropna()
        path_a = df_a_path.split('-')[-1].split('.')[0]
        for df_b_path in df_paths:
            df_b = pd.read_csv(df_b_path).dropna()
            path_b = df_b_path.split('-')[-1].split('.')[0]
            j_score = jaccard_similarity_coefficient(df_a, df_b)
            j_dict[df_a_path + df_b_path] = j_score
    j_matrix = np.array(list(j_dict.values())).reshape(len(df_paths),len(df_paths))
    return j_matrix