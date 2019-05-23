import os
import pandas as pd
import numpy as np
import random
#import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

from keras.callbacks import EarlyStopping



#Le archivo
filename = 'enlace_small.csv'
df = pd.read_csv(filename, encoding = 'unicode_escape')

# Transforms a panel in long format into batches with time dimension of size "window"
def long_to_batches(window, df_original):
    X_escuelas = []
    y_calif =[]

    school_cct = np.unique(list(df_original['cct']))
    num_schools = len(school_cct)

    for school in range(num_schools):
        df_cct = df_original[df_original['cct']==school_cct[school]]
        tam = len(df_cct)
        if tam > window:
            years_cct = list(df_cct['anyo'])
            ciclo=[]
            elem_ciclo = 0 
            for i in range(tam-window):
                while(elem_ciclo< window): 
                    current_year = years_cct[i+elem_ciclo]
                    df_year = df_cct[df_cct['anyo']==current_year]
                    ciclo.append(list(df_year.drop(['cct','anyo'], axis =1).values)[0])
                    elem_ciclo+=1

                X_escuelas.append(ciclo)
                y_calif.append(list(df_cct[df_cct['anyo']==years_cct[i+elem_ciclo]]['p_esp_std'])[0])

                ciclo = []
                elem_ciclo = 0
                
    return X_escuelas, y_calif
 
    
    
    # perc_train is the remaining percentage
def train_test_val_split(X_all, y_all, perc_test, perc_val):
    X_train = []
    X_test = []
    X_val = []
    y_train =[]
    y_test =[]
    y_val =[]

    random.seed(1996)
    tam = len(y_all)
    for i in range(tam):
        coin_toss = random.randint(0,100)/100
        if(coin_toss<perc_test):
                X_test.append(X_all[i])
                y_test.append(y_all[i])
        elif (coin_toss<perc_test+perc_val):
            X_val.append(X_all[i])
            y_val.append(y_all[i])
        else:
            X_train.append(X_all[i])
            y_train.append(y_all[i])
                

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)
    y_train =  np.array(y_train ) 
    y_test =  np.array(y_test ) 
    y_val = np.array(y_val)
    
    print("train size = ", y_train.shape[0], " test size = ", y_test.shape[0], " validation size = " ,y_val.shape[0])
    return X_train, y_train, X_test, y_test,  X_val,  y_val
         
    
def get_model(numUnits, numDropout, valOptimizer):
    model = Sequential() # The LSTM architecture
    model.add(LSTM(units=numUnits)) # First LSTM layer with Dropout regularisation
    model.add(Dropout(numDropout))
    model.add(Dense(units=1))  # The output layer
    model.compile(optimizer=valOptimizer,loss='mean_squared_error')    # Compiling the RNN
    return model


def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    return rmse

def get_test_loss(model, numPatience, numBatch, X_train, y_train, X_test, y_test,  X_val,  y_val):
    es_bc = EarlyStopping(monitor='val_loss', patience=numPatience, verbose=1, mode='auto')
    model.fit(X_train,y_train,epochs=100,batch_size=numBatch, validation_data=(X_val, y_val),callbacks =[es_bc])
    grade_prediction = regressor.predict(X_test)
    loss = return_rmse(y_test,grade_prediction )
    return loss



##Window size 2
windowSize = [2,3,4,5]
units = [19,38,76,100]
dropout = [0,0.1,0.2,0.4]
batch = [32, 64, 128]
optimizers = ['rmsprop', 'adam', 'adagrad']

w2 ={}

for valWindow in windowSize:
    X_escuelas, y_calif = long_to_batches(valWindow, df4)
    X_train, y_train, X_test, y_test,  X_val,  y_val =  train_test_val_split(X_escuelas, y_calif, 0.1,0.1)
    for valOpti in optimizers:
        for numDrop in dropout:
            for numUnits in units:
                model = get_model(numUnits, numDrop,valOpti)
                for numBatch in batch:
                    loss = get_test_loss(model, 4, numBatch, X_train, y_train, X_test, y_test,  X_val,  y_val)
                    nom = "units:"+str(numUnits) +" dropout:"+str(numDrop)+ " optimizer:"+str(valOpti) + " batch:" +str(numBatch)
                    print(nom, "loss = ", loss)
                    if nom in w2:
                        w2[nom].append(loss)
                    else:
                        w2[nom]= [loss]
                        
p2 = pd.DataFrame.from_dict(w2,orient='index')
p2.to_csv("gridsearch_lstm2.csv")
    



