import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import time
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
import torch.optim as optimizers

import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torchtext
from torchtext.data import get_tokenizer
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings('ignore')

#set random seed
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
set_seed(123)


#read data_files
train = pd.read_csv('../input/signate-stcup-2020/train.csv')
test = pd.read_csv('../input/signate-stcup2020-new/test.csv')
submission = pd.read_csv('../input/signate-stcup2020-new/submit_sample.csv')

#define dataset
class Text_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.transform = transform
        
        if self.csv_file.shape[1] == 3:
            #train.shape=>(_,3)
            self.is_train = True
        else:
            #test.shape=>(_,2)
            self.is_train = False
    
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        
        if self.is_train:
            #label returns "jobflag" as onehot vector when you set csv_file="train"
            label = torch.eye(4)[self.csv_file.jobflag.iloc[idx]-1]
        else:
            #label returns the "id" when you set csv_file="test"
            label = self.csv_file.id.iloc[idx]
            
        text = self.csv_file.description.iloc[idx]
            
        if self.transform:
            text = self.transform(text)
        
        del idx
        return text, label
    
#define transform
class BERT_Tokenize(object):
    def __init__(self, model_type, max_len):
        self.max_len = max_len
        
        if model_type == "BERT" or model_type == "TAPTBERT":
            from transformers import BertTokenizer, BertForSequenceClassification
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            
        elif model_type == "ALBERT":
            from transformers import AlbertTokenizer, AlbertForSequenceClassification
            self.bert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            
        elif model_type == "XLNET":
            from transformers import XLNetTokenizer, XLNetForSequenceClassification
            self.bert_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        
        elif model_type == "ROBERTA":
            from transformers import RobertaTokenizer, RobertaForSequenceClassification
            self.bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        elif model_type == "csROBERTA":
            from transformers import AutoTokenizer, AutoModel
            self.bert_tokenizer = AutoTokenizer.from_pretrained("allenai/cs_roberta_base")
            
        elif model_type == "ELECTRA":
            from transformers import ElectraTokenizer, ElectraForSequenceClassification
            self.bert_tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
            
    
    def __call__(self,text):
        inputs = self.bert_tokenizer.encode_plus(
                        text,                       # Sentence to encode.
                        add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                        max_length = self.max_len,  # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,  # Construct attn. masks.
                   )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        del text, inputs
        return torch.LongTensor(ids), torch.LongTensor(mask)

    
#define BERT based model
class BERT_Net(nn.Module):
    def __init__(self, model_type, num_classes):
        super().__init__()
        
        self.model_type = True if model_type == "csROBERTA" or model_type == "TAPTBERT" else False
        
        if model_type == "ALBERT":
            from transformers import AlbertTokenizer, AlbertForSequenceClassification
            self.base_model = AlbertForSequenceClassification.from_pretrained(
                "albert-base-v2",early_stopping=False,num_labels=num_classes)
            
        elif model_type == "BERT":
            from transformers import BertTokenizer, BertForSequenceClassification
            self.base_model = BertForSequenceClassification.from_pretrained(
                "bert-base-cased",early_stopping=False,num_labels=num_classes)
            
        elif model_type == "XLNET":
            from transformers import XLNetTokenizer, XLNetForSequenceClassification
            self.base_model = XLNetForSequenceClassification.from_pretrained(
                "xlnet-base-cased",early_stopping=False,num_labels=num_classes)
            
        elif model_type == "ROBERTA":
            from transformers import RobertaTokenizer, RobertaForSequenceClassification
            self.base_model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",early_stopping=False,num_labels=num_classes)
            
        elif model_type == "csROBERTA":
            from transformers import AutoTokenizer, AutoModel
            self.base_model = AutoModel.from_pretrained("allenai/cs_roberta_base")
            self.classifier = nn.Sequential(
                nn.Linear(768, 768), nn.ReLU(), nn.Dropout(p=0.1),
                nn.Linear(768, 768), nn.ReLU(), nn.Dropout(p=0.1),
                nn.Linear(768, num_classes))
            
        elif model_type == "ELECTRA":
            from transformers import ElectraTokenizer, ElectraForSequenceClassification
            self.base_model = ElectraForSequenceClassification.from_pretrained(
                "google/electra-base-discriminator", num_labels=num_classes)
        
        elif model_type == "TAPTBERT":
            from transformers import AutoModel, AutoConfig
            config = AutoConfig.from_pretrained("../input/tapt-v2/config.json")
            self.base_model = AutoModel.from_pretrained("../input/tapt-v2/pytorch_model.bin", config=config)
            self.classifier = nn.Sequential(
                nn.Linear(768, 768), nn.ReLU(), nn.Dropout(p=0.1),
                nn.Linear(768, 768), nn.ReLU(), nn.Dropout(p=0.1),
                nn.Linear(768, num_classes))
        
    def forward(self, x):
        
        ids, mask = x

        if self.model_type:
            x = self.base_model(input_ids=ids, attention_mask=mask)
            x = self.classifier(x[1])
            preds = x
        else:
            x = self.base_model(input_ids=ids, attention_mask=mask, labels=None)
            preds = x[0]
            
        preds = nn.Softmax(dim=1)(preds)
        return preds
    
def model_config(model_type, version, max_len, class_type=None,):
    # define transforms
    transform = transforms.Compose([
        BERT_Tokenize(model_type=model_type,max_len=max_len)
    ])
    # define model
    if class_type:
        num_classes = 2
        version = str(version) + "_class{}".format(class_type)
    else:
        num_classes = 4
        
    model = BERT_Net(model_type=model_type, 
                     num_classes=num_classes)
    
    return transform, model

#load the saved model
def load_model(model,PATH):
    try:
        checkpoint = torch.load(PATH)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint["accuracy"]
        print("check point epoch: {}, check point loss: {}, accuracy: {}".format(epoch,loss,acc))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("pre-weight is available")

    except:
        print("pre-weight is ''not'' available")

def RUN_TEST(model,test_dataloader,class_type=None):
    
    if class_type:
        num_classes = 2
    else:
        num_classes = 4
        
    probas = np.empty((0,num_classes))
    predictions = torch.LongTensor([])
    ids = torch.LongTensor([])
    
    #test phase
    if __name__ == '__main__':

        model.eval()#####evaluation phase#####
        with torch.no_grad():
            for x, id_number in tqdm(test_dataloader):
                x[0], x[1] = x[0].to(device), x[1].to(device)
                proba = model(x)
                if class_type:
                    preds = torch.argmax(proba,dim=1)
                else:
                    preds = torch.argmax(proba,dim=1)+1
                preds = preds.to("cpu")
                probas = np.append(probas,proba.to('cpu').numpy(),axis=0)
                predictions = torch.cat((predictions,preds),dim=0)
                ids = torch.cat((ids,id_number),dim=0)
                
        if class_type:
            columns = ["0",class_type]
            jobflag = str(class_type)
            title = "job: " + str(class_type)
        else:
            columns = ["1","2","3","4"]
            jobflag = "jobflag"
            title = "prediction\n1: Data Scientist,2: ML Engineer,3: Software Engineer,4: Consultant"
        
        # proba dataframe
        proba_df = pd.DataFrame(data=probas, columns=columns)
        proba_df["ids"] = ids.tolist()
        proba_df.head(3)
        
        #prediction dataframe
        prediction_df = pd.DataFrame({"id": ids.tolist(),
                                      "jobflag": predictions.tolist()
                                     })
        prediction_df.head(3)
        
        
        #visualize the data
        plt.figure(figsize=(20,8))
        # train data
        plt.subplot(1,2,1)
        plt.title("train\n1: Data Scientist,2: ML Engineer,3: Software Engineer,4: Consultant")
        train.jobflag.value_counts().plot(kind="bar")
        #test data
        plt.subplot(1,2,2)
        plt.title(title)
        prediction_df.jobflag.value_counts().plot(kind="bar")
        
        return prediction_df, proba_df
    
def RUN_Prediction(csv_file, model_path, ensemble=False, pseudo_labeling_threshold=None,
                   max_len=128, batch_size=16, num_workers=4):
    
    if ensemble:
        #prepare for ensenble
        probas_df = pd.DataFrame([])
        ensemble_PATH = ""
        n_stack = len(model_path)
        COUNT = 0
    
    for model_path in model_path:

        # read model_path to identify ["model_type", "class_type", "n_folds"]
        save_PATH = model_path.split("/")[-1].strip(".pt")
        model_type = save_PATH.split("_")[0]
        print(f'load == {save_PATH} ...')
        version = save_PATH.split("_")[1]
        class_type = None
        if len(save_PATH.split("_")) == 3:
            if "class" in save_PATH:
                class_type = save_PATH.split("_")[2].strip("class")
            else:
                folds = save_PATH.split("_")[2]
        elif len(save_PATH.split("_")) == 4:
            class_type = save_PATH.split("_")[2].strip("class")
            folds = save_PATH.split("_")[3]
        


        # model configuration
        transform, model = model_config(model_type=model_type,
                                        class_type=class_type,
                                        version=version,
                                        max_len=max_len
                                        )
        # load state_dict()
        load_model(model=model, PATH=model_path)

        #device configuration
        model.to(device)

        # test dataset
        test_dataset = Text_Dataset(csv_file=csv_file,transform=transform)
        # test dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=None,
                                            batch_sampler=None, num_workers=num_workers, collate_fn=None,
                                            pin_memory=True, drop_last=False, timeout=0,
                                         )
        if ensemble:
            prediction_df, proba_df = RUN_TEST(model=model,test_dataloader=test_dataloader,class_type=class_type)
            # save prediction csv
            prediction_df.to_csv(save_PATH+"_sub.csv", index=False, header=False)
            # ensenble proba_df
            if COUNT == 0:
                probas_df = proba_df*0
            probas_df += proba_df
            ensemble_PATH += ("_" + save_PATH)
            COUNT = COUNT + 1
            # save ensenbled proba at last
            if COUNT == n_stack:
                probas_df = probas_df/n_stack
                probas_df.to_csv(ensemble_PATH+"_ensemble.csv", index=False)
                if not class_type:
                    prediction_df = probas_df.drop(["ids"],axis=1)
                    prediction_df["jobflag"] = prediction_df.idxmax(axis=1)
                    prediction_df["id"] = probas_df["ids"]
                    prediction_df = prediction_df.drop(["1","2","3","4"],axis=1)
                    prediction_df = prediction_df[["id","jobflag"]]
                    prediction_df.to_csv(ensemble_PATH+"_sub.csv", index=False, header=False)
                
                if pseudo_labeling_threshold:
                    pseudo_index = np.where(probas_df.drop(["ids"],axis=1)>pseudo_labeling_threshold)[0]
                    pseudo_df = prediction_df.reindex(index=pseudo_index)
                    pseudo_df["description"] = test.reindex(index=pseudo_index)["description"]
                    pseudo_df = pseudo_df.loc[:, ['id', 'description', 'jobflag']]
                    pseudo_df.to_csv(ensemble_PATH+'_pseudo.csv', index=False)
                    
                #visualize the data
                plt.figure(figsize=(20,8))
                # train data
                plt.subplot(1,2,1)
                plt.title("train\n1: Data Scientist,2: ML Engineer,3: Software Engineer,4: Consultant")
                train.jobflag.value_counts().plot(kind="bar")
                #test data
                plt.subplot(1,2,2)
                plt.title("ensemble\n1: Data Scientist,2: ML Engineer,3: Software Engineer,4: Consultant")
                prediction_df.jobflag.value_counts().plot(kind="bar")
        
        else:
            # RUN prediction
            prediction_df, proba_df = RUN_TEST(model=model,test_dataloader=test_dataloader,class_type=class_type)
            # save the prediction and proba
            prediction_df.to_csv(save_PATH+"_sub.csv", index=False, header=False)
            proba_df.to_csv(save_PATH+"_proba.csv", index=False)
            if pseudo_labeling_threshold:
                pseudo_index = np.where(proba_df.drop(["ids"],axis=1)>pseudo_labeling_threshold)[0]
                pseudo_df = prediction_df.reindex(index=pseudo_index)
                pseudo_df["description"] = test.reindex(index=pseudo_index)["description"]
                pseudo_df = pseudo_df.loc[:, ['id', 'description', 'jobflag']]
                pseudo_df.to_csv(save_PATH+'_pseudo.csv', index=False)
                
#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### prediction configuration ######

# prediction for "csv_file"
csv_file = test

# model_path => ["path1", "path2", "path3", ... ]
model_path = [
    "../input/bert-v24/ELECTRA_v40_0.pt",
    "../input/bert-v24/ELECTRA_v40_1.pt",
    "../input/bert-v24/ELECTRA_v40_2.pt",
]
ensemble = True

# threshold should be about 0.99 so that pseudo_label is trustable
pseudo_labeling_threshold = None

#
max_len = 128
batch_size = 32
num_workers = 4

RUN_Prediction(csv_file=csv_file, 
               model_path=model_path, 
               ensemble=ensemble,
               pseudo_labeling_threshold=pseudo_labeling_threshold,
               max_len=max_len, 
               batch_size=batch_size, 
               num_workers=num_workers)






