import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers

# define dataset
class Text_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.transform = transform

        if self.csv_file.shape[1] == 3:
            # train.shape=>(_,3)
            self.is_train = True
        else:
            # test.shape=>(_,2)
            self.is_train = False

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):

        if self.is_train:
            # label returns "jobflag" as onehot vector when you set csv_file="train"
            label = torch.eye(4)[self.csv_file.jobflag.iloc[idx] - 1]
        else:
            # label returns the "id" when you set csv_file="test"
            label = self.csv_file.id.iloc[idx]

        text = self.csv_file.description.iloc[idx]

        if self.transform:
            text = self.transform(text)

        del idx
        return text, label


# define transform
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

    def __call__(self, text):
        inputs = self.bert_tokenizer.encode_plus(
            text,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        del text, inputs
        return torch.LongTensor(ids), torch.LongTensor(mask)


# define BERT based model
class BERT_Net(nn.Module):
    def __init__(self, model_type, num_classes):
        super().__init__()

        self.model_type = True if model_type == "csROBERTA" or model_type == "TAPTBERT" else False

        if model_type == "ALBERT":
            from transformers import AlbertTokenizer, AlbertForSequenceClassification
            self.base_model = AlbertForSequenceClassification.from_pretrained(
                "albert-base-v2", early_stopping=False, num_labels=num_classes)

        elif model_type == "BERT":
            from transformers import BertTokenizer, BertForSequenceClassification
            self.base_model = BertForSequenceClassification.from_pretrained(
                "bert-base-cased", early_stopping=False, num_labels=num_classes)

        elif model_type == "XLNET":
            from transformers import XLNetTokenizer, XLNetForSequenceClassification
            self.base_model = XLNetForSequenceClassification.from_pretrained(
                "xlnet-base-cased", early_stopping=False, num_labels=num_classes)

        elif model_type == "ROBERTA":
            from transformers import RobertaTokenizer, RobertaForSequenceClassification
            self.base_model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base", early_stopping=False, num_labels=num_classes)

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
