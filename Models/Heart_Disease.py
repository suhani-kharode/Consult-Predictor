import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle

df=pd.read_csv("../DataSet/Heart_Disease_Dataset.csv")
df.head()

X=df[['cp','trestbps','chol','fbs','restecg','thalach','exang']]
y=df['target']

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

pickle.dump(model,open("../heart.pkl",'wb'))