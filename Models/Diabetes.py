import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle

df1=pd.read_csv("../DataSet/diabetes_prediction_dataset.csv")
df1.head()

df=df1.sample(2000)
df.head()

# Converting the categorical data into numerical format
smoke_hist={'never':0, 'No Info':1, 'former':2, 'current':3, 'not current':4, 'ever':5}
df['smoking_history']=df['smoking_history'].map(smoke_hist)
df.head()

X=df.drop(['gender','diabetes'],axis=1)
y=df['diabetes']

# Splitting the dataset into train & test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# Model Creation
model=RandomForestClassifier()

# Model Training
model.fit(X_train,y_train)

# Prediction
y_pred=model.predict(X_test)

# Evaluation
acc=accuracy_score(y_test,y_pred)
print("Accuracy:",acc)

clf=classification_report(y_test,y_pred)
print("Classification Report:",clf)

cf=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:",cf)

pickle.dump(model,open("../diabetes.pkl",'wb'))