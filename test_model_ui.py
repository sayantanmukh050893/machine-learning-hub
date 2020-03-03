import pickle
from flask import Flask,request,jsonify,json,render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from util import Util
util = Util()
import os

output_model_path = "E:/Study/GitLab/PIMA_INDIAN_DIABETES/output_model/classification_model.pkl"
#index_path = "E:/Study/GitLab/PIMA_INDIAN_DIABETES/template/index.html"
index_path = "E:/Study/GitLab/PIMA_INDIAN_DIABETES/src/templates/index.html"
predict_path = "E:/Study/GitLab/PIMA_INDIAN_DIABETES/template/predict.html"

app = Flask(__name__)

@app.route('/')
#@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/predict_diabetes',methods=["POST"])

def predict_diabetes():
    print("Inside predict function")
    data = request.form.to_dict()
    df = pd.DataFrame.from_dict(data,orient="index").T
    model = pickle.load(open(output_model_path,"rb"))
    pred = model.predict(df)
    result = pred[0]

    if(result==1):
        prediction =  "PATIENT REPORT : DIABETIC"
    else:
        prediction = "PATIENT REPORT : NOT DIABETIC"
    return render_template("result.html",prediction_text=prediction)

if __name__ == "__main__":
    app.run(port=5555,debug=True)
