import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from load_data import *
from sklearn.ensemble import RandomForestRegressor
import sklearn

def predict_test(feature):
    X_train, y_train, X_test, y_test, X, y = get_data(feature)

    scaler_y = sklearn.preprocessing.MinMaxScaler().fit(y)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    # print(y_train)

    scaler_X = sklearn.preprocessing.MinMaxScaler().fit(X)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    # print(len(X_train))
    # print(len(X_test))

    regr = RandomForestRegressor(random_state=42,n_estimators=100, max_features=10, warm_start=False)

    regr.fit(X_train, y_train)

    y_test_pred = regr.predict(X_test)
    y_train_pred = regr.predict(X_train)
    # print(len(y_test_pred))

    # norm
    # standard_scaler = sklearn.preprocessing.StandardScaler().fit(y_test_pred.reshape(-1, 1))
    # out = standard_scaler.fit_transform(y_test_pred.reshape(-1, 1))

    save_scaler=False # minmax

    if save_scaler==True:
        min_max_scaler = sklearn.preprocessing.MinMaxScaler().fit(y_test_pred.reshape(-1, 1))
        out = min_max_scaler.fit_transform(y_test_pred.reshape(-1, 1))
        out = out


    if feature=='f1':
        filename='predicted_time_cost_per_dialogue_d.txt'
    else:
        filename='predicted_time_cost_per_dialogue_dw.txt'
    with open('output/{}'.format(filename), 'w+') as f:
        if save_scaler==True:
            for i in out:
                f.write(str(round(i[0],4))+'\n')
        else:
            for j in y_test_pred:
                f.write(str(round(j, 4)) + '\n')


    error = mean_squared_error(y_test, y_test_pred)

    minerror = np.sqrt(error)
    print('Mean squared error (MSE): %.2f '
          % minerror)
    print('Coefficient of determination: {}'.format(r2_score(y_test, y_test_pred)))


if __name__=="__main__":
    # feature='f1'
    feature='f2'
    predict_test(feature)