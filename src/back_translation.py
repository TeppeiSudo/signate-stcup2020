import pandas as pd
import translators as ts
from multiprocessing import Pool
from tqdm import tqdm
import random
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def translator_constructor(api):
    if api == 'google':
        return ts.google
    elif api == 'bing':
        return ts.bing
    elif api == 'baidu':
        return ts.baidu
    elif api == 'sogou':
        return ts.sogou
    elif api == 'youdao':
        return ts.youdao
    elif api == 'tencent':
        return ts.tencent
    elif api == 'alibaba':
        return ts.alibaba
    else:
        raise NotImplementedError(f'{api} translator is not realised!')


def imap_unordered_bar(func, args, n_processes: int = 48):
    p = Pool(n_processes, maxtasksperchild=100)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in enumerate(p.imap_unordered(func, args)):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def translate(x):
    if len(x) >= 3:
        try:
            res = [x[0], translator_constructor(API)(x[1], 'en', LANG), x[2]]
            return [res[0], translator_constructor(API)(res[1], LANG, "en"), x[2]]
        except:
            print('hello')
            return [x[0], None, x[2]]
    else:
        try:
            res = [x[0], translator_constructor(API)(x[1], 'en', LANG)]
            return [res[0], translator_constructor(API)(res[1], LANG, "en")]
        except:
            return [x[0], None]


def translate_ver2(x):
    if x[2] == 1:
        is_trans = True if random.random() <= 0.8 else False
    elif x[2] == 2:
        is_trans = True if random.random() <= 0.9 else False
    elif x[2] == 3:
        is_trans = True if random.random() <= 0.7 else False
    elif x[2] == 4:
        is_trans = True if random.random() <= 0.8 else False

    if is_trans:
        try:
            res = [x[0], translator_constructor(API)(x[1], 'en', LANG), x[2]]
            return [res[0], translator_constructor(API)(res[1], LANG, "en"), res[2]]
        except:
            return [x[0], None, x[2]]
    else:
        return [x[0], None, x[2]]


def get_new_df(df, ver2=False):
    if not ver2:
        if df.shape[1] >= 3:
            tqdm.pandas('Translation progress')
            df[['id', 'description', 'jobflag']] = imap_unordered_bar(translate,
                                                                      df[['id', 'description', 'jobflag']].values)
        else:
            tqdm.pandas('Translation progress')
            df[['id', 'description']] = imap_unordered_bar(translate, df[['id', 'description']].values)

        df = df.dropna()
        return df

    else:
        if df.shape[1] >= 3:
            tqdm.pandas('Translation progress')
            df[['id', 'description', 'jobflag']] = imap_unordered_bar(translate_ver2,
                                                                      df[['id', 'description', 'jobflag']].values)
        else:
            tqdm.pandas('Translation progress')
            df[['id', 'description']] = imap_unordered_bar(translate_ver2, df[['id', 'description']].values)

        df = df.dropna()
        return df


def main_for_holdout():
    SPLIT_RATE = 0.8
    n_train = int(len(file) * SPLIT_RATE)
    n_val = len(file) - n_train
    train, val_dataset = train_test_split(file, train_size=n_train, test_size=n_val, random_state=123)
    val_dataset.to_csv('stcup2020-val.csv', index=False)

    save_PATH = f'stcup2020-train-{API}-{LANG}-holdout.csv'
    df = get_new_df(train, ver2)
    df.to_csv(save_PATH, index=False)


def main_for_pseudo_kfold():
    skf = StratifiedKFold(nfold, shuffle=True, random_state=124)
    FOLD = 1
    for tr_idx, val_idx in skf.split(file, file.jobflag):
        val = file.iloc[val_idx]
        val.to_csv(f'stcup2020-val-pseudo-{API}-fold{FOLD}.csv')
        save_PATH = f'stcup2020-train-pseudo-{API}-{LANG}-fold{FOLD}.csv'
        df = get_new_df(file.iloc[tr_idx], ver2)
        df.to_csv(save_PATH, index=False)
        FOLD += 1


def main_for_kfold():
    skf = StratifiedKFold(nfold, random_state=123)
    FOLD = 1
    for tr_idx, val_idx in skf.split(file, file.jobflag):
        val = file.iloc[val_idx]
        val.to_csv(f'stcup2020-val-new-{API}-fold{FOLD}.csv')
        save_PATH = f'stcup2020-train-new-{API}-{LANG}-fold{FOLD}.csv'
        df = get_new_df(file.iloc[tr_idx], ver2)
        df.to_csv(save_PATH, index=False)
        FOLD += 1


def main_for_all():
    save_PATH = f'stcup2020-{filename}-{API}-{LANG}-all.csv'
    df = get_new_df(file, ver2)
    df.to_csv(save_PATH, index=False)


##### config #####
languages = ['bg', 'de', 'ar', 'hi', 'sw', 'vi', 'es', 'el']
train = pd.read_csv('../input/stcup2020-new-train-dataset/new_train.csv')
test = pd.read_csv('../input/signate-stcup2020-new/test.csv')
API = 'google'
nfold = 3
pseudo = pd.read_csv('../input/987pseudo/3_BRE_XA98.7_pseudo.csv')
##### %%%%% #####

if __name__ == '__main__':
    for LANG in languages:
        """
        # usage example
        filename = "test"
        file = test
        ver2 = False
        main_for_all()
        """

        """
        # usage example 
        filename = "train"
        file = train
        ver2 = False
        main_for_all()
        """

        """
        # usage example
        filename = "train"
        file = train
        ver2 = False
        main_for_kfold()
        """

        filename = "train"
        ver2 = False
        file = pd.concat([train, pseudo], join='inner').reset_index(drop=True)
        main_for_pseudo_kfold()

        """
        # usage example
        filename = "train"
        ver2 = False
        file = pd.concat([train, pseudo], join='inner').reset_index(drop=True)
        main_for_holdout()
        """