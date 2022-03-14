
Many countries speak Arabic; however, each country has its own dialect, the aim of this project is to
build a model that predicts the dialect given the text with SVM algorithm, Naive Bayes algorithm and Deep Learning.

## Dataset
  I used tweeter posts datasets we can find data in  `extract_data.zip`
  Or fetching data by run `data_fetching.py`

## Preprocessing 
  - Remove entity mentions (eg. '@united')
  - Remove entity emoji (eg. '🌺')
  - Correct errors (eg. '&amp;' to '&')
  
## Evaluation

I compared with three Algorithm:

<center>
  
|            Model                         | Sentiment acc. |
| ---------------------------------------- |:--------------:|
| SVM                                      |      55 %      |
| Naive Bayes                              |      44 %      |
| Deep Learning (Pretrained-model)         |      77 %      |

</center>

In order to reproduce these results, please install the following requirements:  

```bash
pip install -r requirements.txt
```

Then, run app.py to run server :

```bash
python3 app.py
```
