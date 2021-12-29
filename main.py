

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from preprocessing import text_cleaner
from preprocessing import delete_
import pickle
from bs4 import BeautifulSoup
import spacy
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import spacy

model_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model_classifier = pickle.load(open('classifier_chain_rfc.pkl', 'rb'))
mlb = pickle.load(open('multilabel_binarizer.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))


app = Flask(__name__, template_folder='templates', static_folder='templates/static')

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/", methods= ['POST'])
def predict():
    title = request.form.get('title')
    print(title)
    body = request.form.get('body')
    print(body)

    d = {'Title': [title], 'Body': [body]}
    df = pd.DataFrame(data = d)

    df['Body'] = [BeautifulSoup(text,"lxml").get_text() for text in df['Body'].apply(delete_)]
    df['Body'] = df['Body'].progress_apply(lambda x : text_cleaner(x, spacy.load("en_core_web_sm"), ["NOUN", 'VERB']))
    df['Title']= df['Title'].progress_apply(lambda x : text_cleaner(x, spacy.load("en_core_web_sm"), ["NOUN", 'VERB']))
    df['Doc'] =  df['Body'] +  df['Title']


    if request.method == 'POST':

        keyword = model_vectorizer.transform(df['Doc'])
        keyword = pca.transform(keyword.toarray())
        keyword = model_classifier.predict(keyword)
        keyword = mlb.inverse_transform(keyword)

    return render_template('index.html',
                            title = title,
                            body = body,
                            prediction_text=keyword)

if __name__ == '__main__':
    app.run(host="127.0.0.1",port=8080,debug=True)
