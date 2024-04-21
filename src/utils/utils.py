import os
import json
import numpy as np
import pandas as pd


dataPath = r'../resourses/Data/'

def read_data():
    df = pd.DataFrame(columns=['review', 'sentiment'])

    for directory in os.listdir(dataPath):
        if os.path.isdir(dataPath + directory):
            files = np.array(os.listdir(dataPath + directory))
            for file in files:
                with open(os.path.join(dataPath + directory + '/', file), encoding='utf-8') as f:
                    review = f.read()
                    current_df = pd.DataFrame({'review': [review], 'sentiment': directory})
                    df = pd.concat([df, current_df], ignore_index=True)

    return df

def read_config(dir=""):
    with open(dir + 'config.json') as json_file:
        config_data = json.load(json_file)
        return config_data
