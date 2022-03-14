from flask import Flask, request,jsonify, make_response
import pickle
import numpy as np
import joblib
from datapreprocessing import text_preprocessing
from NN_predict import ml_predictor, CustomAraBERTModel

app = Flask(__name__, template_folder = 'template')

def dialect_predictor(text, model_name):
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

@app.route('/modern_nn/', methods = ['POST'])
def model_NN():
    # ml_predictor()
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            text = request_data['tweet']
            
            result = ml_predictor(text)
            return make_response(jsonify({'dialect':result}), 200)
        except:
            return make_response({
                "error":"please check the parameters",
            },400)

@app.route('/classic_ml/', methods = ['POST'])
def classic_Ml():
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            alg_name = request_data['alg_name']
            text = request_data['tweet']
            if alg_name=="" or alg_name=="svm":
                result = dialect_predictor(list(text),'svm_model')
                print(result)
            elif alg_name=="naive":
                result = dialect_predictor(text,'naive_bayes_model')

            return make_response(jsonify({'dialect':result}), 200)
        except:
            return make_response({
                "error":"please check the parameters",
            },400)



if __name__ == '__main__':
    app.run(port=5001)