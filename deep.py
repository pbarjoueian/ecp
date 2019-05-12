# Import required packages

import glob
import random
import datetime
import os, fnmatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from sklearn.externals import joblib

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.models import model_from_json


#%matplotlib inline

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

def create_model(target):
    # Load the dataset
    df = pd.read_csv("dataset/dbBills_cleaned.csv")

    df = df.drop(df.columns[[0, 1]], 1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df = df.fillna(0)

    # Print top 5 reeocrds of dataset
    df.head()

    features = ['xCycleCode', 'xFamilyNum', 'xFaze', 'xAmper', 'xRegionName_Roustaei',
                'xRegionName_Shahri', 'xUsageGroupName_Keshavarzi', 'xUsageGroupName_Khanegi',
                'xUsageGroupName_Omoomi', 'xUsageGroupName_Sanati', 'xUsageGroupName_Sayer',
                'xBakhshCode_1', 'xBakhshCode_2', 'xBakhshCode_4',
                'xTimeControlCode_1', 'xTimeControlCode_2', 'xTimeControlCode_3',
                'xTariffOldCode_1010', 'xTariffOldCode_1011', 'xTariffOldCode_1110',
                'xTariffOldCode_1111', 'xTariffOldCode_1990', 'xTariffOldCode_2110',
                'xTariffOldCode_2210', 'xTariffOldCode_2310', 'xTariffOldCode_2410',
                'xTariffOldCode_2510', 'xTariffOldCode_2610', 'xTariffOldCode_2710',
                'xTariffOldCode_2990', 'xTariffOldCode_2992', 'xTariffOldCode_3110',
                'xTariffOldCode_3210', 'xTariffOldCode_3310', 'xTariffOldCode_3410', 
                'xTariffOldCode_3520', 'xTariffOldCode_3540', 'xTariffOldCode_3740', 
                'xTariffOldCode_3991', 'xTariffOldCode_4410', 'xTariffOldCode_4610', 
                'xTariffOldCode_4990', 'xTariffOldCode_5110', 'xTariffOldCode_5990',
                'days_difference', 'month']

    X = df[features]
    X = np.matrix(X.values.tolist())
    y = df[target]
    
    #Variables
    y=y.values.reshape(-1,1)
#     y=np.reshape(y, (-1,1))
    scaler = MinMaxScaler()
    print(scaler.fit(X))
    print(scaler.fit(y))
    xscale=scaler.transform(X)
    yscale=scaler.transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)
        
    model = Sequential()
    model.add(Dense(12, input_dim=46, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)
    
    # evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("{0}.json".format(target), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{0}.h5".format(target))
    print("Saved model to disk")

create_model("mediumDailyUsage")
create_model("highDailyUsage")
create_model("lowDailyUsage")
