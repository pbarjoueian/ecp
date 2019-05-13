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
from sklearn import svm

import joblib

def create_model(xTrain, xTest, yTrain, yTest, name):
    clf = svm.SVR(C=12.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
        kernel='rbf', max_iter=-1, shrinking=True,
        tol=0.001, gamma='auto', verbose=False)
    clf.fit(xTrain, yTrain)

    print(mean_squared_error(yTest, clf.predict(xTest)))

    joblib.dump(clf, '../models/SVR-{0}.pkl'.format(name))


def main():
    df = pd.read_csv("../dataset/dbBills_cleaned.csv")

    df = df.drop(df.columns[[0]], axis=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df = df.fillna(0)

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
    y_medium = df["mediumDailyUsage"]
    y_high = df["highDailyUsage"]
    y_low = df["lowDailyUsage"]

    xTrain, xTest, yTrain, yTest = train_test_split(X, y_medium, test_size = 0.1, random_state = 0)
    create_model(xTrain, xTest, yTrain, yTest, 'Medium')
    print("Medium SVM Model has been created!")
    xTrain, xTest, yTrain, yTest = train_test_split(X, y_high, test_size = 0.1, random_state = 0)
    create_model(xTrain, xTest, yTrain, yTest, 'High')
    print("High SVM Model has been created!")
    xTrain, xTest, yTrain, yTest = train_test_split(X, y_low, test_size = 0.1, random_state = 0)create_model(xTrain, xTest, yTrain, yTest)
    create_model(xTrain, xTest, yTrain, yTest, 'Low')
    print("Low SVM Model has been created!")

if __name__ == "__main__":
    main()