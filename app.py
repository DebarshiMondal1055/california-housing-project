import pickle

from flask import Flask, request, jsonify,url_for,render_template

import numpy as np

app=Flask(__name__)

model=pickle.load(open('linear_regression_model.pkl','rb'))

scaler=pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=[float(x) for x in request.form.values()]
    #convert data into numpy array
    final_input_values=np.array(data).reshape(1,-1) #-1 detects number of rows automatically
    
    #transform the input data
    
    scaled_input=scaler.transform(final_input_values)
    
    model_prediction=model.predict(scaled_input)
    return jsonify(model_prediction[0])


if __name__=='__main__':
    app.run(debug=True)