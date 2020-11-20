import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score

# create a scorer object (in this case mse since we're doing regression)
mse = make_scorer(mean_squared_error)

# calculates cv score for model given feature set and label set
# prints and also returns the result

def cv_score_model(model, feature, label, return_results=False):
    '''
    model: sklearn model object
    feature: feature set
    label: label set, index must be same as feature
    return_results: a boolean indicating whether just to print or to return results as well
    '''
    # democrat modeling with cv
    cv_error1 = cross_val_score(model, feature, label.Democrat,cv=10,scoring=mse)

    print("mean cross-validation error for democrat is: ", np.mean(cv_error1))
    print("rmse for democrat is around: ", np.sqrt(np.mean(cv_error1)))

    # republican modeling with cv
    cv_error2 = cross_val_score(model, feature, label.Republican,cv=10,scoring=mse)

    print("mean cross-validation error for republican is: ", np.mean(cv_error2))
    print("rmse for republican is around: ", np.sqrt(np.mean(cv_error2)))

    # cv error for predicting the difference 
    cv_error3 = cross_val_score(model, feature, label.Republican-label.Democrat,cv=10,scoring=mse)

    print("mean cross-validation error for difference is: ", np.mean(cv_error3))
    print("rmse for difference is around: ", np.sqrt(np.mean(cv_error3)))
    if return_results:
        return cv_error1, cv_error2, cv_error3

    