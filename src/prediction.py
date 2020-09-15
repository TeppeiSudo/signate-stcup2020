import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import warnings

from models.utils import set_seed
from models.datamodule import Text_Dataset, BERT_Tokenize, BERT_Net

# utils configuration
warnings.filterwarnings('ignore')
set_seed(123)

# read data_files
train = pd.read_csv('../input/signate-stcup-2020/train.csv')
test = pd.read_csv('../input/signate-stcup2020-new/test.csv')
submission = pd.read_csv('../input/signate-stcup2020-new/submit_sample.csv')


def model_config(model_type, version, max_len, class_type=None, ):
    # define transforms
    transform = transforms.Compose([
        BERT_Tokenize(model_type=model_type, max_len=max_len)
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


# load the saved model
def load_model(model, PATH):
    try:
        checkpoint = torch.load(PATH)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint["accuracy"]
        print("check point epoch: {}, check point loss: {}, accuracy: {}".format(epoch, loss, acc))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("pre-weight is available")

    except:
        print("pre-weight is ''not'' available")


def RUN_TEST(model, test_dataloader, class_type=None):
    if class_type:
        num_classes = 2
    else:
        num_classes = 4

    probas = np.empty((0, num_classes))
    predictions = torch.LongTensor([])
    ids = torch.LongTensor([])

    # test phase
    model.eval()  #####evaluation phase#####
    with torch.no_grad():
        for x, id_number in tqdm(test_dataloader):
            x[0], x[1] = x[0].to(device), x[1].to(device)
            proba = model(x)
            if class_type:
                preds = torch.argmax(proba, dim=1)
            else:
                preds = torch.argmax(proba, dim=1) + 1
            preds = preds.to("cpu")
            probas = np.append(probas, proba.to('cpu').numpy(), axis=0)
            predictions = torch.cat((predictions, preds), dim=0)
            ids = torch.cat((ids, id_number), dim=0)

        if class_type:
            columns = ["0", class_type]
            jobflag = str(class_type)
            title = "job: " + str(class_type)
        else:
            columns = ["1", "2", "3", "4"]
            jobflag = "jobflag"
            title = "prediction\n1: Data Scientist,2: ML Engineer,3: Software Engineer,4: Consultant"

        # proba dataframe
        proba_df = pd.DataFrame(data=probas, columns=columns)
        proba_df["ids"] = ids.tolist()
        proba_df.head(3)

        # prediction dataframe
        prediction_df = pd.DataFrame({"id": ids.tolist(),
                                      "jobflag": predictions.tolist()
                                      })
        prediction_df.head(3)

        # visualize the data
        plt.figure(figsize=(20, 8))
        # train data
        plt.subplot(1, 2, 1)
        plt.title("train\n1: Data Scientist,2: ML Engineer,3: Software Engineer,4: Consultant")
        train.jobflag.value_counts().plot(kind="bar")
        # test data
        plt.subplot(1, 2, 2)
        plt.title(title)
        prediction_df.jobflag.value_counts().plot(kind="bar")

        return prediction_df, proba_df




def RUN_Prediction(csv_file, model_path, ensemble=False, pseudo_labeling_threshold=None,
                   max_len=128, batch_size=16, num_workers=4):
    if ensemble:
        # prepare for ensenble
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

        # device configuration
        model.to(device)

        # test dataset
        test_dataset = Text_Dataset(csv_file=csv_file, transform=transform)
        # test dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=None,
                                     batch_sampler=None, num_workers=num_workers, collate_fn=None,
                                     pin_memory=True, drop_last=False, timeout=0,
                                     )
        if ensemble:
            prediction_df, proba_df = RUN_TEST(model=model, test_dataloader=test_dataloader, class_type=class_type)
            # save prediction csv
            prediction_df.to_csv(save_PATH + "_sub.csv", index=False, header=False)
            # ensenble proba_df
            if COUNT == 0:
                probas_df = proba_df * 0
            probas_df += proba_df
            ensemble_PATH += ("_" + save_PATH)
            COUNT = COUNT + 1
            # save ensenbled proba at last
            if COUNT == n_stack:
                probas_df = probas_df / n_stack
                probas_df.to_csv(ensemble_PATH + "_ensemble.csv", index=False)
                if not class_type:
                    prediction_df = probas_df.drop(["ids"], axis=1)
                    prediction_df["jobflag"] = prediction_df.idxmax(axis=1)
                    prediction_df["id"] = probas_df["ids"]
                    prediction_df = prediction_df.drop(["1", "2", "3", "4"], axis=1)
                    prediction_df = prediction_df[["id", "jobflag"]]
                    prediction_df.to_csv(ensemble_PATH + "_sub.csv", index=False, header=False)

                if pseudo_labeling_threshold:
                    pseudo_index = np.where(probas_df.drop(["ids"], axis=1) > pseudo_labeling_threshold)[0]
                    pseudo_df = prediction_df.reindex(index=pseudo_index)
                    pseudo_df["description"] = test.reindex(index=pseudo_index)["description"]
                    pseudo_df = pseudo_df.loc[:, ['id', 'description', 'jobflag']]
                    pseudo_df.to_csv(ensemble_PATH + '_pseudo.csv', index=False)

                # visualize the data
                plt.figure(figsize=(20, 8))
                # train data
                plt.subplot(1, 2, 1)
                plt.title("train\n1: Data Scientist,2: ML Engineer,3: Software Engineer,4: Consultant")
                train.jobflag.value_counts().plot(kind="bar")
                # test data
                plt.subplot(1, 2, 2)
                plt.title("ensemble\n1: Data Scientist,2: ML Engineer,3: Software Engineer,4: Consultant")
                prediction_df.jobflag.value_counts().plot(kind="bar")

        else:
            # RUN prediction
            prediction_df, proba_df = RUN_TEST(model=model, test_dataloader=test_dataloader, class_type=class_type)
            # save the prediction and proba
            prediction_df.to_csv(save_PATH + "_sub.csv", index=False, header=False)
            proba_df.to_csv(save_PATH + "_proba.csv", index=False)
            if pseudo_labeling_threshold:
                pseudo_index = np.where(proba_df.drop(["ids"], axis=1) > pseudo_labeling_threshold)[0]
                pseudo_df = prediction_df.reindex(index=pseudo_index)
                pseudo_df["description"] = test.reindex(index=pseudo_index)["description"]
                pseudo_df = pseudo_df.loc[:, ['id', 'description', 'jobflag']]
                pseudo_df.to_csv(save_PATH + '_pseudo.csv', index=False)


if __name__ == '__main__':

    # device configuration
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
