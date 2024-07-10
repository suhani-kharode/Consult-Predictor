from flask import Flask,render_template,request,redirect,url_for,flash
import pickle
import numpy as np

app=Flask(__name__,template_folder='templates')

@app.route('/') 
def dashboard():
    return render_template("p.html")

@app.route('/heart') 
def heart():
    return render_template("heart.html")

@app.route('/diabetes') 
def diabetes():
    return render_template("Diabetes.html")

@app.route('/kidney') 
def kidney():
    return render_template("Kidney.html")

@app.route('/liver') 
def liver():
    return render_template("Liver.html")

def ValuePredictor1(to_predict_list):
    to_predict=np.array(to_predict_list).reshape(1,7)
    loaded_model=pickle.load(open('heart.pkl','rb'))
    result=loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict_heart',methods=['POST','GET'])
def predict_heart():
    if request.method=='POST':
        to_predict_list=request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list=list(map(int,to_predict_list))
        result=ValuePredictor1(to_predict_list)
        if int(result)==1:
            pred="Please consult a doctor, you are likely to have Heart Disease."
        else:
            pred="No need to worry, you have no chance of getting Heart Disease."
        return render_template("result.html",prediction_text=pred)

def ValuePredictor2(to_predict_list):
    to_predict=np.array(to_predict_list).reshape(1,7)
    loaded_model=pickle.load(open('kidney.pkl','rb'))
    result=loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict_kidney',methods=['POST','GET'])
def predict_kidney():
    if request.method=='POST':
        to_predict_list=request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list=list(map(int,to_predict_list))
        result=ValuePredictor2(to_predict_list)
        if int(result)==1:
            pred="Please consult a doctor, you are likely to have Kidney Disease."
        else:
            pred="No need to worry, you have no chance of getting Kidney Disease."
        return render_template("result.html",prediction_text=pred)

def ValuePredictor3(to_predict_list):
    to_predict=np.array(to_predict_list).reshape(1,7)
    loaded_model=pickle.load(open('diabetes.pkl','rb'))
    result=loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict_diabetes',methods=['POST','GET'])
def predict_diabetes():
    if request.method=='POST':
        to_predict_list=request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        #to_predict_list=list(map(int,to_predict_list))
        result=ValuePredictor3(to_predict_list)
        if int(result)==1:
            pred="Please consult a doctor, you are likely to have Diabetes."
        else:
            pred="No need to worry, you have no chance of getting Diabetes."
        return render_template("result.html",prediction_text=pred)


def ValuePredictor4(to_predict_list):
    to_predict=np.array(to_predict_list).reshape(1,8)
    loaded_model=pickle.load(open('liver.pkl','rb'))
    result=loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict_liver',methods=['POST','GET'])
def predict_liver():
    if request.method=='POST':
        to_predict_list=request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        #to_predict_list=list(map(int,to_predict_list))
        result=ValuePredictor4(to_predict_list)
        if int(result)==1:
            pred="Please consult a doctor, you are likely to have Liver Disease."
        else:
            pred="No need to worry, you have no chance of getting Liver Disease."
        return render_template("result.html",prediction_text=pred)


    
if __name__=="__main__":
    app.run(debug=True)
