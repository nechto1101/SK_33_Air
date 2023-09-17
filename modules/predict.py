import pandas as pd
import dill
import json
import glob
import os
import random
from pathlib import Path
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '..')
# path = os.path.expanduser('~/airflow_hw')


def predict():
    # <YOUR_CODE>
    mod_cars_pipe = sorted(os.listdir(f'{path}/data/models'))
    with open(f'{path}/data/models/{mod_cars_pipe[-1]}', 'rb') as file:
        model = dill.load(file)
    print(f'{mod_cars_pipe[-1]}')
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    files_list = os.listdir(f'{path}/data/test')
    for jsonfile in files_list:
        with open(f'{path}/data/test/{jsonfile}') as file:
            data = json.load(file)
            df = pd.DataFrame.from_dict([data])
            pred = model.predict(df)
            x = {'car_id': df.id, 'pred': pred}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis=0, ignore_index=True)

    df_pred.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    return print(df_pred.to_string(index=False))


if __name__ == '__main__':
    predict()