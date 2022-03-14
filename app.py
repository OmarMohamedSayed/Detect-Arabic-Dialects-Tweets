from flask import Flask, request,jsonify, make_response
import pickle
import numpy as np
import joblib
import unicodedata
import re

app = Flask(__name__, template_folder = 'template')

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

def dialectPredictor(text, model_name):
    clean_text = []
    predict_classes = []
    loaded_model = joblib.load(f'{model_name}.joblib')
    for i in text:
        clean_text.append(text_preprocessing(i))
    result = loaded_model.predict(clean_text)
    class_label = ['AE', 'BH', 'DZ', 'EG', 'IQ', 'JO', 'KW', 'LB', 'LY', 'MA', 'OM', 'PL',
       'QA', 'SA', 'SD', 'SY', 'TN', 'YE']
    
    for i in result:
        predict_classes.append(class_label[i])
    return predict_classes


@app.route('/classic_ml/', methods = ['POST'])
def classicMl():
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            alg_name = request_data['alg_name']
            text = request_data['tweet']
            if alg_name=="" or alg_name=="svm":
                result = dialectPredictor(list(text),'svm_model')
                print(result)
            elif alg_name=="naive":
                result = dialectPredictor(text,'naive_bayes_model')

            return make_response(jsonify({'dialect':result}), 200)
        except:
            return make_response({
                "error":"please check the parameters",
            },400)

if __name__ == '__main__':
    app.run(port=5001)