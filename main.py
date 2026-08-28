import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl','rb'))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=float(request.form.get('bhk'))
    bath=float(request.form.get('bath'))
    sqft=float(request.form.get('total_sqft'))

    # input dataframe as this is how our predict works -->double list
    input_df=pd.DataFrame(
        [[location,sqft,bath,bhk]],
        columns=['location','total_sqft','bath','bhk']
    )
    prediction = pipe.predict(input_df)[0]*1e5
    #it will give us a list so at 0th position we will have our answer, so that's why
    return str(np.round(prediction,2))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))