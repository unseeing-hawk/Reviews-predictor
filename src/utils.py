import os
import numpy as np
import pandas as pd


path = r'resources/Data/'

def read_data():
    df = pd.DataFrame(columns=['reviews', 'sentiment'])
    
    for directory in os.listdir(path):
        if os.path.isdir(os.path.join(path, directory)):
            files = np.array(os.listdir(os.path.join(path, directory)))
            for file in files:
                with open(os.path.join(path, directory, file), encoding='utf-8') as f:
                    review = f.read()
                    current_df = pd.DataFrame({'reviews': [review], 'sentiment': directory})
                    df = pd.concat([df, current_df], ignore_index=True)
    return df
