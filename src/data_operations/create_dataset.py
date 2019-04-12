import os

import pandas as pd


def combine_files_from_dir(directory: str):
    df = pd.DataFrame()
    
    for filename in os.listdir(directory):
        df = pd.concat([df, pd.read_json(os.path.join(directory, filename))])

    return df
