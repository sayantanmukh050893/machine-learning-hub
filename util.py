
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,precision_score,f1_score
from imblearn.combine import SMOTETomek
class Util():

    def read_csv_data(self,file_path):
        return pd.read_csv(file_path)

    def remove_unwanted_column(self,data,column):
        data.drop(columns=[column],axis=1,inplace=True)
        return data

    def show_column_names(self,data):
        columns = data.columns
        print("The dataset contains the following columns {}".format(columns))

    def change_data_type(self,data,column,to_data_type):
        data[column] = data[column].astype(to_data_type)
        return data

    def scale_variables(self,data,scaler):
        columns = data.columns
        for column in columns:
            if((data[column].dtype=="float64") or (data[column].dtype=="int64") or (data[column].dtype=="float")):
                X = np.array(data[column])
                X = X.reshape(-1,1)
                data[column] = scaler.fit_transform(X)
        return data

    def train_test_split(self,data,dependant_variable,test_size):
        y = data[dependant_variable]
        X = data.drop(columns=[dependant_variable],axis=1)
        smote = SMOTETomek()
        X_res, y_res = smote.fit_resample(X,y)
        X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=test_size)
        return X_train,X_test,y_train,y_test

    def train_model(self,model,X_train,y_train):
        model.fit(X_train,y_train)
        return model

    def predict(self,model,X_test):
        pred = model.predict(X_test)
        return pred

    def get_classification_report(self,y_test,pred):
        return classification_report(y_test,pred)

    def model_score(self,model,model_name,X_train,X_test,y_train,y_test):
        model = self.train_model(model,X_train,y_train)
        pred = self.predict(model,X_test)
        accuracy = np.round(accuracy_score(y_test,pred),2)
        precision = np.round(precision_score(y_test,pred),2)
        f1 = np.round(f1_score(y_test,pred),2)
        return print("The performance of {} is : Accuracy : {} Precision : {} F1-score : {}".format(model_name,accuracy,precision,f1))
        #return print(self.get_classification_report(y_test,pred))

    def generalise_data(self,data,column,value):
        data[data[column]>value][column] = value
        #index = data[data[column]>value].index
        #data.iloc[index,column] = value
        return data
