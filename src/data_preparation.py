import os
cwd = os.getcwd()
import glob
import pandas as pd
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

original_data_path = cwd + "/data/original/"
training_data_path = cwd + "/data/train/"
test_data_path = cwd + "/data/test/"

try:
    files = glob.glob(original_data_path+"*.csv")

    for file in files:
        diabetes_data = pd.read_csv(file)
except FileNotFoundError:
    print("Original File not found.")


    diabetes_index = list(diabetes_data.index)
    training_data_num = np.floor(0.80*diabetes_data.shape[0]).astype(int)
    training_data_index = random.sample(diabetes_index,training_data_num)

    diabetes_training_data = diabetes_data.iloc[training_data_index,]
    diabetes_training_data.to_csv(training_data_path+"/training_data.csv")

    test_data_index = [index not in training_data_index for index in diabetes_index]
    diabetes_test_data = diabetes_data.iloc[test_data_index,]
    diabetes_test_data.drop(columns=["Outcome"],axis=1,inplace=True)
    diabetes_test_data.to_csv(test_data_path+"/test_data.csv")
