import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

dropped_columns = ["CD",
                   "State_Full",
                   "State",
                   "State_Id",
                   "Vote_2012",
                   "Vote_2016",
                   "Vote_2020",
                   "Ticket_CD_Demo",
                   "District_Name"]

sc = StandardScaler()

def process_features(df, is_dat_2020=False):
    
    data = df.copy()
    data = data.drop(columns=set(data.columns).intersection(dropped_columns))
    
    divide_by_pop_cols = ["Females",
                      "Veterans",
                      "White_People",
                      "Afr_Am_People",
                      "Asian",
                      "Children",
                      "Less_Highschool",
                      "Bachelor_Holders",
                      "Below_Poverty_Level_LTM",
                      "American_Ind_Alk_Ntv",
                      "Cuban_Origin",
                      "Puerto_Rican_Origin",
                      "Dominican_Origin",
                      "Mexican_Origin",
                      "Native_US",
                      "Foreign_Born",
                      "Labor_Force_Eligible",
                      "Unemployed",
                      "Workers"]

    divide_by_pop_cols = divide_by_pop_cols+list(df.filter(regex="Working").columns)
    
    data[divide_by_pop_cols] = data[divide_by_pop_cols].apply(lambda x: x / data.Pop * 100) 
    
    
    if is_dat_2020:
        feature = pd.DataFrame(sc.fit_transform(data), 
                               columns=data.columns,
                               index=data.index)
        return shuffle(feature)
    
    feature = data.drop(columns=["Democrat","Republican"])
    label = data[["Democrat","Republican"]]
    
    feature = pd.DataFrame(sc.fit_transform(feature), 
                       columns=feature.columns,
                       index=feature.index)

    
    label = data[["Democrat", "Republican"]].copy()

    # shuffle the data to reduce between data points when training and testing
    feature, label = shuffle(feature, label)
    
    return feature, label