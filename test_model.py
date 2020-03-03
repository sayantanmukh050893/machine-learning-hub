import pickle5 as pickle
from flask import Flask,request,jsonify,json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from util import Util
util = Util()
import os

output_model_path = "E:/Study/GitLab/PIMA_INDIAN_DIABETES/output_model/classification_model.pkl"

app = Flask(__name__)

@app.route('/predict-diabetes',methods=["POST"])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame.from_dict(data,orient="index").T
    model = pickle.load(open(output_model_path,"rb"))
    pred = model.predict(df)
    result = pred[0]
    print(pred)
    if(result==1):
        return "PATIENT REPORT : DIABETIC"
    else:
        return "PATIENT REPORT : NOT DIABETIC"

if __name__ == "__main__":
    app.run(port=6000,debug=True)
