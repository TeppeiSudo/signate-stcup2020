# SIGNATE Student Cup 2020
https://signate.jp/competitions/281

SP(15th place)

## transformers
* BERT

* RoBERTa

* ALBERT

* XLNET

* ELECTRA

* csRoBERTA(https://huggingface.co/allenai/cs_roberta_base)

* furthe-pretrained BERT (pretrained by MaskedLM)

(csRoBERTa and further-pretrained BERT were not used for final submission.)

## preprocess

* normalization and lemmatization (used to find duplicates)

* remove the dupulicates
(some data duplicates in only job-description not in job-flag . it may means these datas are "boundary" so I also removed these kind of duplicates.)

## augmentation

* Backtranslation (google trans API)
(it will take a few hours...)

* Easy Data Augmentation (https://github.com/jasonwei20/eda_nlp)

(EDA was not used for final submission because the lower score than BT in CV.)

## postprocess or others

* 3fold-kfold(you should use 4 or more folds but transformers need a tons of memory...)

* ensembling

* pseudo-labeling(it almost didn't work)
