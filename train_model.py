import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from util import Util
from sklearn.preprocessing import StandardScaler
import glob
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
cwd = os.getcwd()
training_data_path = cwd + "/data/train/"
os.chdir(training_data_path)
training_data_path = os.getcwd()

relative_path = "E:/Study/GitLab/PIMA_INDIAN_DIABETES/"

# Fetching all the file fro training the model
training_file = glob.glob(training_data_path+"/*.csv")[0]

# Initializing the object for util class
util = Util()

# Reading the training data from the training data file path prvided
training_data = util.read_csv_data(training_file)

# Removing the unwanted columnsfrom the dataset
training_data = util.remove_unwanted_column(training_data,"Unnamed: 0")

# Showing the column names
util.show_column_names(training_data)

# Changing the data type of Outcome from int64 to category since it is the
# dependant variable and it is a classification problem

training_data = util.change_data_type(training_data,"Outcome","category")

#print(training_data.describe())

#print(training_data.corr())
#sns.heatmap(training_data.corr())
#plt.show()

#sns.pairplot(training_data)
#plt.show()

#fig = plt.figure(figsize=(10,10))

#print(training_data.shape)

# plt.figure(figsize=(8,20))
# for i,j in enumerate(training_data.columns[0:8]):
#     plt.subplot(4,2,i+1)
#     plt.boxplot(training_data[j])
#     #plt.xlabel("Feature Name {}".format(j))
#     plt.ylabel(j)
#     plt.tight_layout()
# plt.show()

# Generalising outliers of Pregancies
training_data = util.generalise_data(training_data,"Pregnancies",5)

# Generalising outliers of BloodPressure
training_data = util.generalise_data(training_data,"BloodPressure",120)

# Generalising outliers of Insulin
training_data = util.generalise_data(training_data,"Insulin",500)

# Generalising outliers of DiabetesPedigreeFunction
training_data = util.generalise_data(training_data,"DiabetesPedigreeFunction",1.75)

# Generalising outliers of BMI
training_data = util.generalise_data(training_data,"BMI",50)

# Scaling the independant variables
#scaler = StandardScaler()
#training_data = util.scale_variables(training_data,scaler)
#print(training_data.head())
#print(training_data["Outcome"].value_counts())
X_train,X_test,y_train,y_test = util.train_test_split(training_data,"Outcome",test_size=0.20)

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
util.model_score(classifier,"neural network model",X_train,X_test,y_train,y_test)

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
util.model_score(logistic_model,"logistic model",X_train,X_test,y_train,y_test)

from sklearn.naive_bayes import BernoulliNB
naive_bayes = BernoulliNB()
util.model_score(naive_bayes,"naive bayes model",X_train,X_test,y_train,y_test)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
util.model_score(decision_tree,"decision tree model",X_train,X_test,y_train,y_test)

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
util.model_score(random_forest,"random forest model",X_train,X_test,y_train,y_test)

from sklearn.svm import SVC
svc_model = SVC()
util.model_score(svc_model,"svc model",X_train,X_test,y_train,y_test)

from sklearn.ensemble import GradientBoostingClassifier
gradient_classifier = GradientBoostingClassifier()
util.model_score(gradient_classifier,"gradient boosting",X_train,X_test,y_train,y_test)

from sklearn.ensemble import AdaBoostClassifier
ada_boost_classifier = AdaBoostClassifier()
util.model_score(ada_boost_classifier,"ada boost boosting",X_train,X_test,y_train,y_test)

from sklearn.ensemble import VotingClassifier

print("------------------------------------------------------------------------------")

vote1 = VotingClassifier(estimators=[('neural_classifier',classifier),('logistic_model',logistic_model)],weights=[2,1],voting='hard')
util.model_score(vote1,"neural network and logistic models",X_train,X_test,y_train,y_test)

vote2 = VotingClassifier(estimators=[('naive_bayes',naive_bayes),('logistic_model',logistic_model)],weights=[1,2],voting='hard')
util.model_score(vote2,"logistic and navie bayes models",X_train,X_test,y_train,y_test)

vote3 = VotingClassifier(estimators=[('decision_tree',decision_tree),('naive_bayes',naive_bayes)],weights=[2,1],voting='hard')
util.model_score(vote3,"decision and navie bayes models",X_train,X_test,y_train,y_test)

vote4 = VotingClassifier(estimators=[('decision_tree',decision_tree),('random_forest',random_forest)],weights=[1,2],voting='hard')
util.model_score(vote4,"decision and random forest models",X_train,X_test,y_train,y_test)

vote5 = VotingClassifier(estimators=[('random_forest',random_forest),('svc_model',svc_model)],weights=[2,1],voting='hard')
util.model_score(vote5,"decision and random forest models",X_train,X_test,y_train,y_test)

vote6 = VotingClassifier(estimators=[('ada_boost_classifier',ada_boost_classifier),('gradient_classifier',gradient_classifier)],weights=[1,1],voting='hard')
util.model_score(vote6,"ada  boost and gradient boost models",X_train,X_test,y_train,y_test)

vote7 = VotingClassifier(estimators=[('vote1',vote1),('vote2',vote2),('vote3',vote3),('vote4',vote4),('vote5',vote5),('vote6',vote6)],weights=[1,1,1,1,1,1],voting='hard')
util.model_score(vote7,"final model",X_train,X_test,y_train,y_test)

os.chdir(relative_path+"/output_model/")
model_dump_dir = os.getcwd()+"/classification_model.pkl"
pickle.dump(vote7,open(model_dump_dir,"wb"))
