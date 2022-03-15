
Many countries speak Arabic; however, each country has its own dialect, the aim of this project is to
build a model that predicts the dialect given the text with SVM algorithm, Naive Bayes algorithm and Deep Learning.

## Dataset
  I used tweeter posts datasets we can find data in  two files:
    
    - the first file (tweets with ids):`extract_data.zip`
    - the second file (ids with dialect): `dialect_dataset.csv`
    
  Or fetching data by run `data_fetching.py`

## Preprocessing 
  - Merge two datasets on ids
  - Remove entity mentions (eg. '@united')
  - Remove entity emoji (eg. '🌺')
  - Correct errors (eg. '&amp;' to '&')
  - Create dataset loader class

## Run
```bash
pip install -r requirements.txt
```

Then, run datapreprocessing.py to fetching data
```bash
python3 datapreprocessing.py
```
`
usage: data_fetching.py [-h] [--file-path FILE_PATH]
                        [--request-ids-limit REQUEST_IDS_LIMIT] [--url URL]
                        [--output-file-name OUTPUT_FILE_NAME] [--debug]
`

Then, run model.py to create train models and save it 
(you can download pretrained models:
  - SVM model `https://drive.google.com/file/d/1P9-RXla2nWyH0MzH9waUYG7KFajMwH9S/view?usp=sharing` 
  - Naive Bayes model `https://drive.google.com/file/d/1-4ml_ORLHvE1dIeoqItrDXWWnTA4xqRe/view?usp=sharing`
  - DeepLearning model `https://drive.google.com/file/d/1-4ml_ORLHvE1dIeoqItrDXWWnTA4xqRe/view?usp=sharing`
  ):
  
```bash
python3 model.py
```

Then, run app.py to run server :

```bash
python3 app.py
```
then, can import postman collection with all requests

```bash
dialect_collection.postman_collection.json
```

## Evaluation

I compared with three Algorithm:

<center>
  
|            Model                         | Sentiment acc. |
| ---------------------------------------- |:--------------:|
| SVM                                      |      55 %      |
| Naive Bayes                              |      44 %      |
| Deep Learning (Pretrained-model)         |      77 %      |

</center>
