import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def welcome():
    print("Welcome to Salary Prediction System")
    print("Press enter key to procees")
    input()
def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    for x in content_list:
        if(x.split('.')[-1]=='csv'):
            csv_files.append(x)
    if(len(csv_files)==0):
        return "No csv files in the directory"
    else:
        return csv_files
def display_and_select(csv_files):
    i=0
    print("File number  File name")
    for file_names in csv_files:
         print(i,"......",file_names)
         i+=1
    file_name=int(input("Enter the file no: you wan't to select"))
    return csv_files[file_name]
def graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred):
    plt.scatter(X_train,Y_train,color='red',label='training data')
    plt.plot(X_train,regressionObject.predict(X_train),color='blue',label='Best fit')
    plt.scatter(X_test,Y_test,color='green',label='test data')
    plt.scatter(X_test,Y_pred,color='black',label='Predicted data')
    plt.title("Salary vs Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("salary")
    plt.legend()
    plt.show()
def main():
    welcome()
    try:
        csv_files=checkcsv()
        if(csv_files=="No csv files in the directory"):
            raise FileNotFoundError("No csv files in the directory")
        print("Reading csv file data")
        time.sleep(5)
        csv_file=display_and_select(csv_files)
        print("File Selected Successfully")
        print("-----------------------------")
        print("Creating Dataset")
        dataset=pd.read_csv(csv_file)
        print("Creation successfull")
        X=dataset.iloc[:,:-1].values
        Y=dataset.iloc[:,-1].values
        s=float(input("Enter test data size(between 0 and 1)"))
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=s)#return numpy format data
        print("Model creation in Progress")
        regressionObject=LinearRegression()
        regressionObject.fit(X_train,Y_train)
        print("Model Created")
        print("Press Enter Key TO Predict Data in trained model")
        input()
        Y_pred=regressionObject.predict(X_test)
        print("X_test....Y_test....Y_train")
        i=0
        while(i<len(X_test)):
            print(X_test[i],"....",Y_test[i],"....",Y_pred[i])
            i+=1
        print("Press Enter key to see above data in graphical format")
        input()
        graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred)
        r2=r2_score(Y_test,Y_pred)
        print("Our model is %2.2f%% accurate" %(r2*100))
        
        print("Now you can predict salary of an employee using model")
        print("Enter experience in years of candidate ,seperated by comma")
        exp=[float(e) for e in input().split(',')]
        ex=[]
        for x in exp:
            ex.append([x])
        experience=np.array(ex)
        salaries=regressionObject.predict(experience)#pass only numpy in predict functio

        plt.scatter(experience,salaries,color='black')
        plt.plot(experience,salaries,color='red')
        plt.xlabel("Years of experience")
        plt.ylabel("Salaries")
        plt.show()

        print("In TABULAR FORMAT")
        d=pd.DataFrame({'Experience':exp,"salaries":salaries})
        print(d) 
        
        
    except FileNotFoundError:
        print("No csv files in the directory")
        print("Press Enter key to exit")
        input()
        exit()

if(__name__=="__main__"):
    main()
    input()
