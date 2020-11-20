from cv_score_model import *
import numpy as np
import pandas as pd

dat12 = pd.read_csv("./data_2012.csv").iloc[:,1:]
dat16 = pd.read_csv("./data_2016.csv").iloc[:,1:]
dat12 = dat12.set_index("CD_ID")
dat16 = dat16.set_index("CD_ID")

def validation(model, dropped_columns=[],
               preprocessor=False, return_results=False):
    '''
    compares cv error and test error by getting cv error for 2012 and then test error for 2016
    dat_2012: 2012 data
    dat_2016: 2016 data
    model: sklearn model object
    dropped_columns: defining any dropped column names
    preprocessor: define any preprocessor steps with an iterable
    return_results: boolean to actually return results. default False
    '''
    # set up the data
    val_train_X = dat12.drop(columns=dropped_columns+["Democrat","Republican"])
    val_train_Y = dat12[["Democrat","Republican"]] 

    val_test_X = dat16.drop(columns=dropped_columns+["Democrat","Republican"])
    val_test_Y = dat16[["Democrat","Republican"]] 

    # apply preprocessor
    if preprocessor:
        for process in preprocessor:
            val_train_X = process.fit_transform(val_train_X)
            val_test_X = process.fit_transform(val_test_X)

    # train xgb model on val_train_X and val_train_Y and get cv error
    cv_score_model(model,val_train_X, val_train_Y)
    
    print("\n")
    # predict on the test set on democrats
    model.fit(val_train_X, val_train_Y.Democrat)
    val_test_error = mse(model,val_test_X, val_test_Y.Democrat)
    print("test error on democrat on 2016 data is: ", val_test_error)
    print("root test error on democrat on 2016 data is: ", np.sqrt(val_test_error))
    
    # predict on the test set on republicans
    model.fit(val_train_X, val_train_Y.Republican)    
    val_test_error = mse(model,val_test_X, val_test_Y.Republican)
    print("test error on republican on 2016 data is: ", val_test_error)
    print("root test error on republican on 2016 data is: ", np.sqrt(val_test_error))

    # predict on the test set on democrats - republicans
    model.fit(val_train_X, val_train_Y.Democrat - val_train_Y.Republican)
    val_test_error = mse(model,val_test_X, val_test_Y.Democrat - val_test_Y.Republican)
    print("test error on democrat on 2016 data is: ", val_test_error)
    print("root test error on difference on 2016 data is: ", np.sqrt(val_test_error))

