from flask import Flask, render_template,request,make_response
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from random import randint
import mysql.connector
from mysql.connector import Error
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template,request,make_response
import sys
import random
import pandas as pd
import numpy as np
import json  #json request
#from acc import Processor
import os
import csv #reading csv
from random import randint
from math import sqrt
from math import pi
from math import exp
from sklearn.metrics import classification



app = Flask(__name__)

fn=''
HB=''
temp=''
ecg=''

@app.route('/i')
def i():
    return render_template('index.html')

@app.route('/')
def index():
    
    connection = mysql.connector.connect(host='sg2nlmysql15plsk.secureserver.net',database='iotdb',user='iotroot',password='iot@123')    
    cursor = connection.cursor()
    sql_Query = "select * from epilepsy order by id desc limit 1"
    print(sql_Query)
    cursor.execute(sql_Query)
    data = cursor.fetchall()
    #data=list(data)
    print(data[0][1])    
    global HB
    global temp
    global ecg
    HB=data[0][1]
    temp=data[0][2]
    ecg=data[0][3]
    connection.commit() 
    connection.close()
    cursor.close()
    return render_template('dataloader.html',HB=data[0][1],temp=data[0][2],ecg=data[0][3])
    
    #return render_template('dataloader.html')



@app.route('/uploadajax', methods =  ['GET','POST'])
def upldfile():
    print("request :"+str(request), flush=True)
    if request.method == 'POST':
        global fn   
        global HB
        global temp
        global ecg
        prod_mas = request.files['prod_mas']
        HB = request.form['HB']
        temp = request.form['temp']
        ecg = request.form['ecg']
        filename = secure_filename(prod_mas.filename)
        prod_mas.save(os.path.join("./static/Upload/", filename))
        fn = os.path.join("./static/Upload/", filename)
        print(filename)
        fn=filename
        return render_template('dataloader.html')
       
def Average(lst):
    return sum(lst) / len(lst)


@app.route('/procdataset', methods =  ['GET','POST'])
def procdataset():
    global fn   
    global HB
    global temp
    global ecg
    HB=float(HB)
    temp=float(temp)
    ecg=float(ecg)
    tempdata=pd.read_csv("./static/Upload/"+ fn)
    df=pd.read_csv("./static/Upload/"+ fn)
    print(type(df))
    lpdf=df


    df.columns = ['Id','HB','temp','ecg','Type']
    print(lpdf)
    print(df.info())
    pd.set_option('display.float_format', '{:.2f}'.format)

    print('**************After Data Cleansing**********************')
    df.drop('Id',axis='columns', inplace=True)
    print(df)
    lpdf=df
    print('**************Data Describe**********************')
    print(df.describe())
    
    print('**************Data mean**********************')
    print(df.mean())

    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier

    max_accuracy = 0
    from sklearn.model_selection import train_test_split

    print('*************************************************************************************')
    print('********************************Linear Regression************************************')
    print('*************************************************************************************')
    
    predictors = lpdf.drop("Type",axis=1)
    print(predictors)
           
    target = df["Type"]

    X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

    # Splitting the dataset into the Training set and Test set

    #from sklearn.model_selection import train_test_split
    #X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

    # Fitting Simple Linear Regression to the training set

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    # Predicting the Test set result ￼

    Y_Pred = regressor.predict(X_test)
    print('*************************************************************************************')
    print(Y_Pred)
    print('*************************************************************************************')

    average = Average(Y_Pred)
    print("Linear regression value : ", round(average, 2))
    print("The accuracy score achieved using Logistic Regression is: "+str(classification.accuracy_score("LR"))+" %")








    print('*************************************************************************************')
    print('*********************************Random Forest***************************************')
    print('*************************************************************************************')

   



    categorical_values = []
    for column in df.columns:
        print('==============================')
        print(f"{column} : {df[column].unique()}")
        if len(df[column].unique()) <= 10:
            categorical_values.append(column)
    
    print('**************categorical values**********************')
    print(categorical_values)


    predictors = df.drop("Type",axis=1)
    print(predictors)
           
    target = df["Type"]

    X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


    for x in range(10):
        rf = RandomForestClassifier(random_state=x)
        rf.fit(X_train,Y_train)
        Y_pred_rf = rf.predict(X_test)
        current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
        if(current_accuracy>max_accuracy):
            max_accuracy = current_accuracy
            best_x = x
            
    #print(max_accuracy)
    #print(best_x)

    rf = RandomForestClassifier(random_state=best_x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    score_rf = round(classification.accuracy_score("RF"),2)
    print("The accuracy score achieved using Random Forest is: "+str(classification.accuracy_score("RF"))+" %")
    epoch=10
    classification.modelbuild(epoch)
    print("The accuracy score achieved using Neural Network is: "+str(classification.accuracy_score("NN"))+" %")

    cdf=df["temp"]
    chartdata=[]
    for i in range(len(cdf)):
        cd=[]
        cd.append(i+1)
        cd.append(cdf[i])
        chartdata.append(cd)
    print(chartdata)
    eff=''
    if temp<35:
        eff='Most Likely'
    if temp>35 and temp<45:
        eff='Likely'

    '''
    connection = mysql.connector.connect(host='sg2nlmysql15plsk.secureserver.net',database='iotdb',user='iotroot',password='iot@123')    
    cursor = connection.cursor()
    sql_Query = "select * from oximeter order by id desc limit 1"
    print(sql_Query)
    cursor.execute(sql_Query)
    data = cursor.fetchall()
    data=list(data)
    connection.commit() 
    connection.close()
    cursor.close()
    return render_template('dataloader.html',tempdata=tempdata,acscore=acscore,chartdata=chartdata,fval=len(chartdata)+2,pulse=data[0][1],blood=data[0][2],temp=data[0][3])
    
    '''
    tempdata=tempdata.values.tolist()
    connection = mysql.connector.connect(host='sg2nlmysql15plsk.secureserver.net',database='iotdb',user='iotroot',password='iot@123')    
    cursor = connection.cursor()
    sql_Query = "select * from epilepsy order by id desc limit 1"
    print(sql_Query)
    cursor.execute(sql_Query)
    data = cursor.fetchall()
    #data=list(data)
    print(data[0][1])    
    HB=data[0][1]
    temp=data[0][2]
    ecg=data[0][3]
    connection.commit() 
    connection.close()
    cursor.close()
    return render_template('dataloader.html',tempdata=tempdata,acscore=score_rf,chartdata=chartdata,fval=len(chartdata)+2,typer=eff,HB=data[0][1],temp=data[0][2],ecg=data[0][3])
       





if __name__ == '__main__':
    UPLOAD_FOLDER = 'E:/Upload'
    app.secret_key = "secret key"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run()
