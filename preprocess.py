#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 10:21:47 2020

@author: horst
"""

import os
import numpy as np
import pandas as pd

import json
from matplotlib import path

from sklearn.preprocessing import OneHotEncoder

# dictionary of the files to be downloaded
month_url = {'april_data' :'uber-raw-data-apr14.csv', 'may_data' :'uber-raw-data-may14.csv',
            'june_data' :'uber-raw-data-jun14.csv', 'july_data' :'uber-raw-data-jul14.csv',
            'august_data' :'uber-raw-data-aug14.csv', 'september_data' :'uber-raw-data-sep14.csv'}

# path to store the downloaded data
SAVING_PATH = os.path.join('datasets')


data_dict = {}


def load_csv_data(csv_path):
    return pd.read_csv(csv_path)


def load_data(month):
    csv_filename = month_url[month]
    csv_path = os.path.join(SAVING_PATH, csv_filename)
    pd_filename = month
    data_dict.update({pd_filename : load_csv_data(csv_path)})
    
    return data_dict[month]
    
    
def convert_neighborhood(x_data):
    geofile = json.load(open("NY_neighborhoods.geojson"))
    geo_points = list(zip(x_data['Lon'], x_data['Lat']))
    
    for feature in geofile['features']:
        coords = feature['geometry']['coordinates'][0]
        p = path.Path(coords)
        inds = p.contains_points(geo_points)
        list_neighborhoods = [str(feature['properties']['neighborhood'])]*np.sum(inds)
        x_data.loc[x_data.index[inds], 'Neighborhood'] = list_neighborhoods
        
        return x_data
    
def preprocess_month(month):
    # load the data and generate a DataFrame
    x = load_data(month)

    # Add the label attribute to the data
    x['Pickups'] = 1
    
    # Add new parameter Neighborhood to cluster the longitude and latitude into spatial clusters
    x_add = x.copy()
    x_add['Neighborhood'] = np.zeros(len(x_add))
    x_add = convert_neighborhood(x_add)
    
    # Remove all non-matching entries
    x_add = x_add[x_add['Neighborhood'] != 0]
    
    
    # Convert Date/Time into a time series format
    x_timeseries = x_add.copy()
    x_timeseries.index = pd.to_datetime(x_timeseries['Date/Time'])
    x_timeseries.sort_index(inplace=True)
    x_timeseries.drop(labels=['Date/Time'], axis=1, inplace=True)
    
    
    # Delete unnecessary parameters
    x_neighbor = x_timeseries.drop(['Lat', 'Lon', 'Base'], axis = 1)
    
    
    # Cluster the data within time intervals of one hour
    x_cluster = x_neighbor.resample('H').agg({'Pickups' : 'sum', 'Neighborhood': 'nunique'})
    
    # Extend the data by weekdays, weekend check and hours of the day
    x_cluster['Weekday'] = x_cluster.index.weekday
    x_cluster['Is_weekend'] = x_cluster.index.map(lambda x: 1 if x.weekday() > 4 else 0)
    x_cluster['Hour_of_day'] = x_cluster.index.hour
    
    # Split labels from training set
    X_train = x_cluster.drop('Pickups', axis = 1)
    y_train = x_cluster['Pickups'].copy()
    
    # Categorical attributes
    map_dict_weekday = {0: "Mon", 1: "Tue", 2: "Wen", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    X_train['Day_of_week'] = X_train['Weekday'].map(map_dict_weekday)
    X_train.drop(labels=['Weekday'], axis=1, inplace=True)
    
    map_dict_hour = {0: "H_1", 1: "H_2", 2: "H_3", 3: "H_4", 4: "H_5", 5: "H_6", 6: "H_7", 7: "H_8",
                8: "H_9", 9: "H_10", 10: "H_11", 11: "H_12", 12: "H_13", 13: "H_14", 14: "H_15", 15: "H_16",
                16: "H_17", 17: "H_18", 18: "H_19", 19: "H_20", 20: "H_21", 21: "H_22", 22: "H_23", 23: "H_24"}
                
    X_train['Hours'] = X_train['Hour_of_day'].map(map_dict_hour)
    X_train.drop(labels=['Hour_of_day'], axis=1, inplace=True)
    
    X_train_prep = X_train[['Is_weekend', 'Day_of_week', 'Hours']].copy()
    X_train_prep.reset_index(drop=True)
    
    # One-Hot-Encoder
    one_hot_encoder = OneHotEncoder(sparse=False)
    X_train_cat_days = X_train_prep[['Day_of_week']]
    X_train_cat_days_1hot = one_hot_encoder.fit_transform(X_train_cat_days)
    X_train_cat_days_1hot
    
    X_train_cat_hours = X_train_prep[['Hours']]
    X_train_cat_hours_1hot = one_hot_encoder.fit_transform(X_train_cat_hours)
    X_train_cat_hours_1hot
    
    Is_weekend_array = np.expand_dims(np.asarray(X_train_prep['Is_weekend']), axis=1)
    
    result = np.concatenate((Is_weekend_array, X_train_cat_days_1hot, X_train_cat_hours_1hot), axis=1)
    
    X_train_prep = pd.DataFrame(result)
    
    return X_train_prep, y_train
