import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

# Loading the model
model=pickle.load(open('knnmodel.pkl','rb'))
scale=pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']   # The input is given in json format and it will be captured and stored in data variable 
    # The data will be in key-value pairs 
    print(data)
    arrdata=np.array(list(data.values()))
    arrdata.astype(float)
    #intdata=int(arrdata)
   # print(np.array(list(data.values())).reshape(1,-1))
    data1=arrdata.reshape(1,-1)
    newscaled_data=scale.transform(data1)
    output=model.predict(newscaled_data)
    print(int(output[0]))
    return jsonify(int(output[0]))
    


@app.route('/predict',methods=['POST'])
def predict():
    data1=[float(x) for x in request.form.values()]
    data1=np.array(data1).reshape(1,-1)
    final_input=scale.transform(data1)
    print(final_input)
    output=model.predict(final_input)[0]
    if output==1:
         return render_template("home.html",prediction_text="The passenger is survived")
    else:
        return render_template("home.html",prediction_text="The person is not survived")

# @app.route('/predict',methods=['POST'])
# def predict():
#     data=[float(x) for x in request.form.values()]
#     final_input=scale.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output=model.predict(final_input)[0]
#     return render_template("home.html",prediction_text="The House price prediction is {}".format(output))




if __name__=="__main__":
        app.run(debug=True)




