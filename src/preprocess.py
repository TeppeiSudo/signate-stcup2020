import pandas as pd
from models.utils import split_sentences, drop_duplicated_sentence, split_sentences_for_test

# read data files
train = pd.read_csv('../input/signate-stcup-2020/train.csv')
test = pd.read_csv('../input/signate-stcup2020-new/test.csv')
submission = pd.read_csv('../input/signate-stcup2020-new/submit_sample.csv')

# preprocess for train dataset
clean_and_evil, evil, clean = drop_duplicated_sentence(train)
additional_train = split_sentences(clean)
train = pd.concat([clean, additional_train])
train = train.reset_index(drop=True)
_, _, train = drop_duplicated_sentence(train)
train = train.reset_index(drop=True)
train.to_csv('new_train.csv')

# preprocess for test dataset
additional_test = split_sentences_for_test(test)
test = pd.concat([test, additional_test])
test.to_csv('test_ex.csv', index=False)