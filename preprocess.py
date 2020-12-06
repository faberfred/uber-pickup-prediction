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
    
    # Convert Date/Time into a time series format
    x_date = x['Date/Time']
    x_date = pd.to_datetime(x_date)
    x_data = x_date.to_frame()
    x['Date/Time'] = x_data['Date/Time']
    
    # Add new parameter Neighborhood to cluster the longitude and latitude into spatial clusters
    x_add = x.copy()
    x_add['Neighborhood'] = np.zeros(len(x_add))
    
    x_add = convert_neighborhood(x_add)
    
    # Remove all non-matching entries
    x_add = x_add[x_add['Neighborhood'] != 0]
    x_neighbor = x_add.drop(['Lat', 'Lon', 'Base'], axis = 1)
    
    # Cluster the data within time intervals of one hour
    x_cluster = x_neighbor.resample('H', on='Date/Time').agg({'Pickups' : 'sum', 'Neighborhood': 'nunique'})
    
    # Split labels from training set
    X = x_cluster.drop('Pickups', axis = 1)
    y = x_cluster['Pickups'].copy()
    
    return X, y
