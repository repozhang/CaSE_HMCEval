from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from module_2_HEE.hee.__utils__.csvutil import *
import numpy as np
from module_2_HEE.hee.__utils__.my_readability import *
from sklearn import datasets, linear_model
import sklearn
from module_2_HEE.hee.__utils__.mape_cal import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from module_2_HEE.hee.model.load_data import *
import time

# Load data
def cross_validation(feature):

    X_train,Y_train,X_test,Y_test,X,y=get_data(feature)

    myfoldnumber=5

    kf=KFold(n_splits=myfoldnumber, random_state=397, shuffle=True)
    i=0
    newscore_dict={}
    rlist=[[],[],[],[],[],[],[],[],[],[]]
    for train_index, test_index in kf.split(X):
        i+=1
        if i in range(1,myfoldnumber+1):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test=X.iloc[list(train_index)], X.iloc[list(test_index)]
            # print(X_train,X_test)
            y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]


            """<transform to 0 and 1>"""
            scaler_y = sklearn.preprocessing.MinMaxScaler().fit(y)
            y_train = scaler_y.transform(y_train)
            y_test = scaler_y.transform(y_test)

            scaler_X = sklearn.preprocessing.MinMaxScaler().fit(X)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)
            """<transform to 0 and 1>"""

            # regr = RandomForestRegressor(random_state=397,n_estimators=100, max_features=10,warm_start=False)
            regr = RandomForestRegressor(random_state=397,n_estimators=100, max_features=8,warm_start=False)
            print('RandomForestRegressor')

            # Train the model using the training sets
            regr.fit(X_train, y_train)

            y_test_pred = regr.predict(X_test)
            # print(len(y_test_pred),len(y_test))
            y_train_pred = regr.predict(X_train)
            print(y_test_pred)

            msetest = mean_squared_error(y_test, y_test_pred)
            msetrain = mean_squared_error(y_train, y_train_pred)

            rmsetest = np.sqrt(msetest)
            rmsetrain = np.sqrt(msetrain)

            maetest=mean_absolute_error(y_test, y_test_pred)
            maetrain=mean_absolute_error(y_train, y_train_pred)

            r2_test = r2_score(y_test, y_test_pred)
            r2_train = r2_score(y_train, y_train_pred)

            mape_test= mean_absolute_percentage_error(y_test, y_test_pred)
            mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

            rlist[0].append(msetest)
            rlist[1].append(msetrain)

            rlist[2].append(rmsetest)
            rlist[3].append(rmsetrain)

            rlist[4].append(maetest)
            rlist[5].append(maetrain)

            rlist[6].append(mape_test)
            rlist[7].append(mape_train)

            rlist[8].append(r2_test)
            rlist[9].append(r2_train)

            # print('coefficient', regr.coef_ )
            print('MSE,RMSE,MAE,MAPE,R2:{},{},{},{},{},{},{},{},{},{}'.format(msetest,msetrain,
                                                    rmsetest,rmsetrain,
                                                    maetest,maetrain,
                                                    mape_test,mape_train,
                                                    r2_test,r2_train))

    # cross-validation avg
    namelist=['TEST-MSE','TRAIN-MSE','TEST-RMSE','TRAIN-RMSE','TEST-MAE','TRAIN-MAE','TEST-MAPE','TRAIN-MAPE','TEST-R2','TRAIN-R2']
    for k,name in zip(range(0,10),namelist):
        print(name,round(np.mean(rlist[k]),4))


feature='f1'
start = time.time()
cross_validation(feature)
end = time.time()
print(time, end-start)
