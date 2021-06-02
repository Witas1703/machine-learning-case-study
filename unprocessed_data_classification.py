import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import metrics

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


df = pd.read_csv("data/weatherAUS_original_data.csv")
df = df.drop("Date",axis = 1)

# encoding RainTomorrow and RainToday as binary values
df.RainToday.replace(("Yes", "No"), (1,0), inplace = True)
df.RainTomorrow.replace(("Yes", "No"), (1,0), inplace = True)

df.dropna(inplace = True)

# print(df.shape)

# columns to be changed to one-hot encoding
categorical_columns = ["WindGustDir", "WindDir9am", "WindDir3pm", "Location"]

# creating one-hot encoding
df = pd.get_dummies(df, columns = categorical_columns)

# print(df.head())

# print(df.describe())

print(df['RainTomorrow'].sum()/len(df))

y = df.RainTomorrow.to_numpy()
X = df.drop(columns=['RainTomorrow']).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

pipe = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k = 20)) ,
        ('classifier', SVC())
    ], 
    verbose=True
    ) 


pipe.fit(X_train, y_train)

y_predicted = pipe.predict(X_test)
print(metrics.balanced_accuracy_score(y_test, y_predicted))

report = metrics.classification_report(y_test, y_predicted)
print(report)
print("Accuracy of the model is:",metrics.accuracy_score(y_test,y_predicted)*100,"%")
cm = metrics.confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True,cmap="YlGnBu")
plt.title("Confusion Matrix for model")
plt.show()