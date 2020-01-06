from flask import Flask
from flask import Flask,send_from_directory, render_template,Response,request ,make_response, session   
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
from werkzeug.utils import secure_filename
from datetime import datetime, date, timedelta
import random
import string
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import gc
import time
import lightgbm as lgb
import random

app = Flask(__name__,static_folder = "templates", template_folder='templates')

@app.route('/')
def index():
	return render_template('/index.html')

@app.route('/train', methods = ['POST', 'GET'])
def train():
    if request.method == 'POST':
        results = request.form.to_dict()
        if results['username'] == "abhi123" and results['password'] == "zzz":
            # %% [code]
            if len(results) == 2:
                return render_template('/train.html',showlogin=0,message="",results=results)
            
            f = request.files['fileToUpload']
            filePath = f.filename
            print(filePath)
            f.save(secure_filename(filePath))
            # %% [code]
            train = pd.read_csv(filePath) 

            # %% [code]
            train['year'] = train['Date'].str[:4]  #This will get the first four letters from train..Nothing nothing
            train['month'] = train['Date'].str[5:7]
            train['day'] = train['Date'].str[8:10]

            # %% [code]
            train = train.drop('Date',axis=1)

            # %% [code]
            train['windDirSame'] = 0 
            train.loc[train['WindDir9am']==train['WindDir3pm'],'windDirSame'] = 1

            # %% [code]
            #Wind Speed, Humidity, Pressure, Cloud, Temperature 
            train['windDiff'] = train['WindSpeed9am'] - train['WindSpeed3pm']
            train['humidityDiff'] = train['Humidity9am'] - train['Humidity3pm']
            train['pressureDiff'] = train['Pressure9am'] - train['Pressure3pm']
            train['CloudDiff'] = train['Cloud9am'] - train['Cloud3pm']
            train['TempDiff'] = train['Temp9am'] - train['Temp3pm']


            train['RainTomorrow'] = train['RainTomorrow'].astype('category').cat.codes


            train.fillna(0)

            train['Location'] = train['Location'].astype('category')
            train['WindGustDir'] = train['WindGustDir'].astype('category')
            train['WindDir9am'] = train['WindDir9am'].astype('category')
            train['WindDir3pm'] = train['WindDir3pm'].astype('category')
            train['RainToday'] = train['RainToday'].astype('category')
            train['year'] = train['year'].astype('category')
            train['month'] = train['month'].astype('category')
            train['day'] = train['day'].astype('category')

            meanRain = train.loc[train['RISK_MM'] != 0]['RISK_MM'].mean()
            medianRain = train.loc[train['RISK_MM'] != 0]['RISK_MM'].median()


            # %% [code]
            #Let's divide the rain into 4 categories

            #If Rain is above mean, then it will be high
            #If the rain is between median and mean, it will be medium
            #If the rain is below median but above 0 then light

            #Make a new column named rain Category
            train['rainCategory'] = 0
            #If rain is light, assign it 1
            train.loc[ (train['RISK_MM'] != 0) & (train['RISK_MM'] < medianRain)  ,'rainCategory'] = 1
            #If rain is medium, assign it 2
            train.loc[ (train['RISK_MM'] > medianRain) & (train['RISK_MM'] < meanRain)  ,'rainCategory'] = 2
            #If rain is high, assign it 3
            train.loc[train['RISK_MM'] > meanRain  ,'rainCategory'] = 3

            # %% [code]
            X_train = train.loc[(train['year'] != '2016') & (train['year'] !=  '2017')]

            test = train.loc[(train['year'] == '2016') | (train['year'] == '2017')]

            # %% [code]
            y = X_train['rainCategory']
            X_train = X_train.drop(['RainTomorrow','RISK_MM','rainCategory'],axis=1)

            y_test = test['rainCategory'] 
            test = test.drop(['RainTomorrow','RISK_MM','rainCategory'],axis=1)

            # %% [code]
            train_set = lgb.Dataset(X_train, label=y)
            valid_set = lgb.Dataset(test, label=y_test)

            # %% [code]
            params = { 
                    "objective" : "multiclass", 
                    "metric" : "multi_error",  
                    "learning_rate" : 0.09,
                    'num_class':4
            }

            model = lgb.train(  params, 
                                train_set = train_set, 
                                num_boost_round=100000, 
                                early_stopping_rounds=200, 
                                verbose_eval=100, 
                                valid_sets=[train_set,valid_set]
                              )
            model.save_model('model.txt')
            return render_template('/index.html',showlogin=0,message="")
        else:
            return render_template('/train.html',showlogin=1,message="Wrong Username or Password")
    else:
        return render_template('/train.html',showlogin=1,message="")






@app.route('/predict.html', methods = ['POST', 'GET'])
def predict():
    if request.method == 'POST':
        results = request.form.to_dict()
        for result in results:
            if results[result] == "":
                results[result] = 0
            
            try:
                results[result] = float(results[result])
            except:
                pass
                

        if results['Rainfall'] > 1:
            results['RainToday'] = "Yes"
        else:
            results['RainToday'] = "No"

        results['Date'] = str(date.today().strftime("%Y-%m-%d"))
        
        
        
       
        df = pd.DataFrame([{'Date': results['Date'] }])
        df['Location'] = results['Location']
        df['MinTemp'] = results['MinTemp']
        df['MaxTemp'] = results['MaxTemp']
        df['Rainfall'] = results['Rainfall']
        df['Evaporation'] = results['Evaporation']
        df['Sunshine'] = results['Sunshine']
        df['WindGustDir'] = results['WindGustDir']
        df['WindGustSpeed'] = results['WindGustSpeed']
        df['WindDir9am'] = results['WindDir9am']
        df['WindDir3pm'] = results['WindDir3pm']
        df['WindSpeed9am'] = results['WindSpeed9am']
        df['WindSpeed3pm'] = results['WindSpeed3pm']
        df['Humidity9am'] = results['Humidity9am']
        df['Humidity3pm'] = results['Humidity3pm']
        df['Pressure9am'] = results['Pressure9am']
        df['Pressure3pm'] = results['Pressure3pm']
        df['Cloud9am'] = results['Cloud9am']
        df['Cloud3pm'] = results['Cloud3pm']
        df['Temp9am'] = results['Temp9am']
        df['Temp3pm'] = results['Temp3pm']
        df['RainToday'] = results['RainToday']
        

        
        
        df['year'] = df['Date'].str[:4]  #This will get the first four letters from train..Nothing nothing
        df['month'] = df['Date'].str[5:7]
        df['day'] = df['Date'].str[8:10]
        df = df.drop('Date',axis=1)
        df['windDirSame'] = 0 
        
        
        
        df.loc[df['WindDir9am']==df['WindDir3pm'],'windDirSame'] = 1
        #Wind Speed, Humidity, Pressure, Cloud, Temperature 
        df['windDiff'] = df['WindSpeed9am'] - df['WindSpeed3pm']
        df['humidityDiff'] = df['Humidity9am'] - df['Humidity3pm']
        df['pressureDiff'] = df['Pressure9am'] - df['Pressure3pm']
        df['CloudDiff'] = df['Cloud9am'] - df['Cloud3pm']
        df['TempDiff'] = df['Temp9am'] - df['Temp3pm']

        df['Location'] = df['Location'].astype('category')
        df['WindGustDir'] = df['WindGustDir'].astype('category')
        df['WindDir9am'] = df['WindDir9am'].astype('category')
        df['WindDir3pm'] = df['WindDir3pm'].astype('category')
        df['RainToday'] = df['RainToday'].astype('category')
        df['year'] = df['year'].astype('category')
        df['month'] = df['month'].astype('category')
        df['day'] = df['day'].astype('category')

        model = lgb.Booster(model_file='model.txt')
        y_pred = model.predict(df)
        y_pred[0][0] = round(round(y_pred[0][0]*100,2)/100,2)
        y_pred[0][1] = round(round(y_pred[0][1]*100,2)/100,2)
        y_pred[0][2] = round(round(y_pred[0][2]*100,2)/100,2)
        y_pred[0][3] = round(round(y_pred[0][3]*100,2)/100,2)
        print(y_pred)
        return render_template('/result.html',pred=y_pred[0])
    else:
        return render_template('/predict.html')

@app.route('/visualization.html', methods = ['POST', 'GET'])
def visualization():
    if request.method == 'POST':
        results = request.form.to_dict()
        df = pd.read_csv('weatherAUS.csv')
        df['year'] = df['Date'].str[:4]
        df['month'] = df['Date'].str[5:7]
        df['day'] = df['Date'].str[8:10]
        df['month'] = df['month'].map({
        '01':'Jan',
        '02':'Feb',
        '03':'Mar',
        '04':'Apr',
        '05':'May',
        '06':'Jun',
        '07':'Jul',
        '08':'Aug',
        '09':'Sep',
        '10':'Oct',
        '11':'Nov',
        '12':'Dec',
        })
        if results['year'] == "All Years" and results['month'] == "No Month":
            print("CASE 1 ==================")
            df2 = df.loc[df.Location == results['Location']]
            piechart = []
            piechart.append(df2.loc[df2.RISK_MM == 0].shape[0])
            piechart.append(df2.loc[(df2.RISK_MM != 0) & (df2.RISK_MM < 2.2)].shape[0])
            piechart.append(df2.loc[(df2.RISK_MM >= 2.2) & (df2.RISK_MM < 6.566875342359574)].shape[0])
            piechart.append(df2.loc[df2.RISK_MM >= 6.566875342359574].shape[0])
            yearlyRainfall = df2.groupby('year')['RISK_MM'].agg('mean')
            yearlyRainfall.plot.bar(x="year",y="RISK_MM", fontsize = 22,figsize=(23,12),legend=False)
            plt.suptitle('Mean Yearly Rainfall in '+results['Location'], fontsize=30)
            plt.xlabel('Year', fontsize=30)
            plt.ylabel('Rainfall in MM', fontsize=30)
            plt.savefig('templates/yearlyRainfall.png')
            fig = plt.gcf()
            plt.close(fig) 
            return render_template('/visualization.html',noData=0,case=1,location=results['Location'],piechart=piechart,randNum=random.randint(0, 99999999999))
        
        elif results['year'] != "All Years" and results['month'] == "No Month":
            print("CASE 2 ==================")
            df2 = df.loc[df.Location == results['Location']]
            df2 = df2.loc[df2.year == results['year']]
            noData = 0
            if df2.shape[0] == 0:
                noData = 1
                return render_template('/visualization.html',noData=noData,case=2,location=results['Location'],year=results['year'],randNum=random.randint(0, 99999999999))

            piechart = []
            piechart.append(df2.loc[df2.RISK_MM == 0].shape[0])
            piechart.append(df2.loc[(df2.RISK_MM != 0) & (df2.RISK_MM < 2.2)].shape[0])
            piechart.append(df2.loc[(df2.RISK_MM >= 2.2) & (df2.RISK_MM < 6.566875342359574)].shape[0])
            piechart.append(df2.loc[df2.RISK_MM >= 6.566875342359574].shape[0])
            yearlyRainfall = df2.groupby('month')['RISK_MM'].agg('mean')
            yearlyRainfall.index = pd.CategoricalIndex(yearlyRainfall.index, categories=['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec'])
            yearlyRainfall = yearlyRainfall.sort_index()
            yearlyRainfall.plot.bar(x="month",y="RISK_MM", fontsize = 22,figsize=(23,12),legend=False)
            plt.suptitle('Mean Monthly Actual Rainfall in '+results['Location'] + ' in ' + results['year'], fontsize=30)
            plt.xlabel('Month', fontsize=30)
            plt.ylabel('Rainfall in MM', fontsize=30)
            plt.savefig('templates/yearlyRainfall.png')
            fig = plt.gcf()
            plt.close(fig)
            return render_template('/visualization.html',noData=noData,case=2,location=results['Location'],year=results['year'],piechart=piechart,randNum=random.randint(0, 99999999999))
        
        elif results['year'] != "All Years" and results['month'] != "No Month":
            print("CASE 3 ==================")
            df2 = df.loc[df.Location == results['Location']]
            df2 = df2.loc[df2.year == results['year']]
            df2 = df2.loc[df2.month == results['month']]
            noData = 0
            if df2.shape[0] == 0:
                noData = 1
                return render_template('/visualization.html',noData=noData,case=3,month=results['month'],location=results['Location'],year=results['year'],randNum=random.randint(0, 99999999999))

            piechart = []
            piechart.append(df2.loc[df2.RISK_MM == 0].shape[0])
            piechart.append(df2.loc[(df2.RISK_MM != 0) & (df2.RISK_MM < 2.2)].shape[0])
            piechart.append(df2.loc[(df2.RISK_MM >= 2.2) & (df2.RISK_MM < 6.566875342359574)].shape[0])
            piechart.append(df2.loc[df2.RISK_MM >= 6.566875342359574].shape[0])
            yearlyRainfall = df2.groupby('day')['RISK_MM'].agg('mean')
            yearlyRainfall.plot.bar(x="day",y="RISK_MM", fontsize = 22,figsize=(23,12),legend=False)
            plt.suptitle('Daily Actual Rainfall in '+results['Location'] + ' in '+ results['month'] + ' ' + results['year'], fontsize=30)
            plt.xlabel('Day', fontsize=30)
            plt.ylabel('Rainfall in MM', fontsize=30)
            plt.savefig('templates/yearlyRainfall.png')
            fig = plt.gcf()
            plt.close(fig)
            return render_template('/visualization.html',noData=noData,case=3,month=results['month'],location=results['Location'],year=results['year'],piechart=piechart,randNum=random.randint(0, 99999999999))
        
        elif results['year'] == "All Years" and results['month'] != "No Month":
            print("CASE 4 ==================")
            df2 = df.loc[df.Location == results['Location']]
            df2 = df2.loc[df2.month == results['month']]
            noData = 0
            if df2.shape[0] == 0:
                noData = 1
                return render_template('/visualization.html',noData=noData,case=4,month=results['month'],location=results['Location'],year=results['year'],randNum=random.randint(0, 99999999999))

            piechart = []
            piechart.append(df2.loc[df2.RISK_MM == 0].shape[0])
            piechart.append(df2.loc[(df2.RISK_MM != 0) & (df2.RISK_MM < 2.2)].shape[0])
            piechart.append(df2.loc[(df2.RISK_MM >= 2.2) & (df2.RISK_MM < 6.566875342359574)].shape[0])
            piechart.append(df2.loc[df2.RISK_MM >= 6.566875342359574].shape[0])
            yearlyRainfall = df2.groupby('year')['RISK_MM'].agg('mean')
            yearlyRainfall.plot.line(x="day",y="RISK_MM", fontsize = 22,figsize=(23,12),legend=False)
            plt.suptitle('Mean Yearly Actual Rainfall in '+results['Location'] + ' in '+ results['month'] + " over the years", fontsize=30)
            plt.xlabel('Year', fontsize=30)
            plt.ylabel('Rainfall in MM', fontsize=30)
            plt.savefig('templates/yearlyRainfall.png')
            fig = plt.gcf()
            plt.close(fig)
            return render_template('/visualization.html',noData=noData,case=4,month=results['month'],location=results['Location'],year=results['year'],piechart=piechart,randNum=random.randint(0, 99999999999))
        
            
        """
        locationdf = df.loc[df.Location == results['Location']]
        yearList = locationdf.year.unique()
        piechartLocation = []
        piechartLocation.append(locationdf.loc[locationdf.RISK_MM == 0].shape[0])
        piechartLocation.append(locationdf.loc[(locationdf.RISK_MM != 0) & (locationdf.RISK_MM < 2.2)].shape[0])
        piechartLocation.append(locationdf.loc[(locationdf.RISK_MM >= 2.2) & (locationdf.RISK_MM < 6.566875342359574)].shape[0])
        piechartLocation.append(locationdf.loc[locationdf.RISK_MM >= 6.566875342359574].shape[0])
        yearlyRainfall = locationdf.groupby('year')['RISK_MM'].agg('mean')
        
        yearlyRainfall.plot.bar(figsize=(23,12), fontsize = 22,legend=False)
        plt.suptitle('Mean Yearly Actual Rainfall in '+results['Location'], fontsize=30)
        plt.xlabel('Year', fontsize=30)
        plt.ylabel('Rainfall in MM', fontsize=30)
        plt.savefig('templates/yearlyRainfall.png')
        """

        return render_template('/visualization.html',location=results['Location'],piechartLocation=piechartLocation,monthlyRainfall=0,Location=results['Location'],yearlyRainfall=1,randNum=random.randint(0, 99999999999),yearList=yearList,totalYears=len(yearList) )
    else:
        return render_template('/visualization.html',yearlyRainfall=0)


if __name__ == "__main__":
    app.run(debug=True,port=8000)