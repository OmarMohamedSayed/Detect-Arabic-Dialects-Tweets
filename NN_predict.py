import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datapreprocessing import text_preprocessing

#load data
dialect_ids_only = pd.read_csv('./dialect_dataset.csv')
dialect_with_title  =  pd.read_json('./output.json')
dialect_data = pd.merge(dialect_with_title,dialect_ids_only, left_on="ids", right_on="id").drop('id', axis=1)
dialect_data.title = dialect_data.title.apply(lambda x: text_preprocessing(x))


tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")

TEXT_MAX_LEN = 70

#Data Loader Class
class MyDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame = dialect_data,
        tokenizer = tokenizer,
        text_max_token_len: int = TEXT_MAX_LEN,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data[index]

        text = data_row['title']

        text_encoding = tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        return dict(
            input_ids=text_encoding['input_ids'].flatten(),
            attention_mask=text_encoding['attention_mask'].flatten(),
        )

#Loading Model
class CustomAraBERTModel(nn.Module):
    def __init__(self):
        super(CustomAraBERTModel, self).__init__()
        self.arabert = AutoModelForSequenceClassification.from_pretrained(
            "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        )
        self.arabert.classifier = nn.Linear(in_features=768, out_features=18, bias=True)

    def forward(self, input_ids, attention_mask):
        output = self.arabert(input_ids, attention_mask=attention_mask)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CustomAraBERTModel()
model =torch.load(Path("./arabert_dialect.pth"), map_location=device)

def ml_predictor(text):
    class_label = ['AE', 'BH', 'DZ', 'EG', 'IQ', 'JO', 'KW', 'LB', 'LY', 'MA', 'OM', 'PL',
       'QA', 'SA', 'SD', 'SY', 'TN', 'YE']
    predict = MyDataset(text, tokenizer)
    predict_classes =[]
    predict_dataloader = DataLoader(predict, shuffle=True, batch_size=1)
    for batch in predict_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch['input_ids'], batch['attention_mask'])
        predict_classes.append(class_label[int(torch.argmax(outputs.logits[0]))])

    return predict_classes

