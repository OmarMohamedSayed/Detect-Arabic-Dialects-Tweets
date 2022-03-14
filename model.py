import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from ar_wordcloud import ArabicWordCloud

from transformers import AutoTokenizer, AutoModelForSequenceClassification
"""**Read Two DataSets**"""

dialect_ids_only = pd.read_csv('./dialect_dataset.csv')
dialect_with_title  =  pd.read_json('./output.json')

dialect_with_title.head()

dialect_ids_only.head()

"""**Merge Two Data Sets (dialect_data_with_posts, dialect_data_with_country) on Ids columns**"""

dialect_data = pd.merge(dialect_with_title,dialect_ids_only, left_on="ids", right_on="id").drop('id', axis=1)
dialect_data.head()

"""**View The Most Repeated Country Dialect**"""

val_count = dialect_data.dialect.value_counts()
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(val_count.index,val_count.values, color ='blue',
        width = 0.4)
 
plt.xlabel("Dialects")

plt.title("Sentiment Data Distribution")
# plt.savefig('Sentiment_Data_Distribution".jpg')
plt.show()

"""**Preprocessing Steps:**


1.   Remove entity mentions (eg. '@united')
2.   Remove entity emoji (eg. '🌺')
3.   Correct errors (eg. '&amp;' to '&')


"""

import unicodedata

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Remove entity emoji (eg. '🌺')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Normalize unicode encoding
    text = unicodedata.normalize('NFC', text)
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    #Remove URLs
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '<URL>', text)
    #Remove Emoji
    text = re.sub(re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030""]+", re.UNICODE),'',text)

    return text

#apply preprocessing on whole data
dialect_data.title = dialect_data.title.apply(lambda x: text_preprocessing(x))
dialect_data

plt.figure(figsize = (20,20)) 
wc = ArabicWordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(dialect_data[dialect_data.dialect == 'EG'].title))
plt.imshow(wc)

"""*   **Remove dialect columns that contain all countries**
*   **Change dialect columns from string to category then save in new column called targets** 
*   **splite data into train & test**



"""

dialects = dialect_data['dialect'].astype('category')
print('categories: {}'.format(dialects.cat.categories.unique()))
targets = {k: v for v, k in enumerate(dialects.cat.categories.values)}
dialect_data['targets'] = dialect_data['dialect'].apply(lambda x : targets[x])
train, test =train_test_split(dialect_data, test_size=0.1, random_state=42)
# check dimintionality of data 
print('dim train: {}'.format(train.shape))
print('dim test: {}'.format(test.shape))
print('y_train description: \n{}'.format(train['targets'].describe()))
print('y_test description: \n{}'.format(test['targets'].describe()))
print('y_train counts: \n{}'.format(train['targets'].value_counts()))
print('y_test counts: \n{}'.format(test['targets'].value_counts()))

"""## Machine Learning Classical Approach
*   SVM
*   MultinomialNB

**Create train_model Function with two encoders**


*   CountVectorizer
*   TfidfTransformer
*   Save Model
"""

import joblib

def train_model(model, data, targets, model_name):
    text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', model),
    ])
    text_clf.fit(data, targets)
    filename = f'{model_name}.joblib'
    joblib.dump(text_clf,filename)
    return text_clf

def get_accuracy(trained_model,X, y):
    predicted = trained_model.predict(X)
    accuracy = np.mean(predicted == y)
    return accuracy

"""## SVM Model """

from sklearn.svm import LinearSVC
trained_clf_SVC = train_model(LinearSVC(), train['title'],train['targets'], 'svm_model')
accuracy = get_accuracy(trained_clf_SVC,test['title'],test['targets'])
print(f"Test dataset accuracy with SVC: {accuracy:.2f}")

"""`**Load Model and Create Predictions**"""

print(test['title'][:6])

print(test['targets'][:6])

loaded_model = joblib.load('svm_model.joblib')
loaded_model.predict(test['title'][:6])

"""## Naive Bayez Model"""

from sklearn.naive_bayes import MultinomialNB
trained_clf_multinomial_nb = train_model(MultinomialNB(), train['title'],train['targets'], 'naive_bayes_model')
accuracy = get_accuracy(trained_clf_multinomial_nb,test['title'],test['targets'])
print(f"Test dataset accuracy with MultinomialNB: {accuracy:.2f}")

"""`**Load Model and Create Predictions**"""

loaded_model = joblib.load('naive_bayes_model.joblib')
loaded_model.predict(test['title'][:6])

"""## Deep Learning Approach (RNN --transfer learning)"""

#load Arabic Tokenizer
tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")

"""**Create Dataset loader class:** 
handle data and send it as batches
"""

TEXT_MAX_LEN = 70

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
        data_row = self.data.iloc[index]

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

        target = data_row['targets']
        
        return dict(
            input_ids=text_encoding['input_ids'].flatten(),
            attention_mask=text_encoding['attention_mask'].flatten(),
            target=torch.tensor(target, dtype=torch.long),
        )

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

#send data to loader class
train_dataset = MyDataset(train, tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128)

test_dataset = MyDataset(test, tokenizer)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=128)

#start training

from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomAraBERTModel()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

num_epochs = 1

num_training_steps = num_epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(batch['input_ids'], batch['attention_mask'])
        logits = outputs.logits
        loss = criterion(logits, batch['target'])
        loss.backward()
        
        optimizer.step()
        
        optimizer.zero_grad()
        progress_bar.update()
    
    torch.save(model, './arabert_dialect.pth')
    print(f'epoch: {epoch} -- loss: {loss}')

"""**RNN Predictions**"""

from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model =torch.load(Path("./arabert_dialect.pth"), map_location=device)
predict = MyDataset(test[0:1], tokenizer)
predict_dataloader = DataLoader(predict, shuffle=True, batch_size=1)


for batch in predict_dataloader:

  batch = {k: v.to(device) for k, v in batch.items()}
        
  outputs = model(batch['input_ids'], batch['attention_mask'])
  print(int(torch.argmax(outputs.logits[0])))
  # prediction = int(torch.max(outputs.logits[0], 1).numpy())
  # prediction

