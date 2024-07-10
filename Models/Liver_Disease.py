import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle

df=pd.read_csv("../DataSet/Liver_patient.csv")
df.head()

# Handling missing values
mean=np.mean(df['Albumin_and_Globulin_Ratio'])
df['Albumin_and_Globulin_Ratio']=df['Albumin_and_Globulin_Ratio'].fillna(mean)

X=df[['Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']]
y=df['Dataset']

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

pickle.dump(model,open("../liver.pkl",'wb'))

