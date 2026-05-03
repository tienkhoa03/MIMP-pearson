import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae
import tsdb
import pandas as pd
from datetime import datetime


def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('Model Size: {:.3f} MB'.format(size_all_mb))
    return size_all_mb

# def load_WiFi_dataset(dataset_name):
#     if dataset_name in ["KDM", "WDS", "LHS"]:
#         ori_floor_df = pd.read_csv(f'./data/fp_sample_{dataset_name}.csv', index_col=0)
#         ori_floor_df = ori_floor_df.sort_values(by=['ts'], ascending=True).reset_index(drop=True)
#         # ori_floor_df = ori_floor_df.sample(frac=1, random_state=2021).reset_index(drop=True)
#
#         # ori_floor_df.to_csv(f'./data/fp_sample_{dataset_name}_sorted.csv', header=True, index=False)
#
#         Feature_len = ori_floor_df.shape[1] - 4
#         print('feature length:', Feature_len)
#         Feature_df = ori_floor_df.iloc[:, :Feature_len]
#         # normalize all samples:
#         locs = ori_floor_df.loc[:, ['x', 'y']].astype('float')
#         mean_x_y = locs.loc[:, ['x', 'y']].mean().values
#         std_x_y = locs.loc[:, ['x', 'y']].std().values
#         base_X_mask = (~np.isnan(Feature_df.values)).astype(int)
#         Feature_df = Feature_df.fillna(-100)
#         mean_f = Feature_df.mean().values
#         std_f = Feature_df.std().values
#         base_X = (Feature_df.values - mean_f) / std_f
#
#         start_time = ori_floor_df.loc[0, 'ts']
#         end_time = ori_floor_df.loc[len(ori_floor_df)-1, 'ts']
#
#         start_time = datetime.fromtimestamp(start_time/1000)
#         end_time = datetime.fromtimestamp(end_time/1000)
#
#         period = (end_time-start_time).total_seconds()/60
#         print('start time:', start_time)
#         print('end time:', end_time)
#         print('period:', period)
#
#     # return base_X


def load_WiFi_dataset(window=6, dataset_name='KDM', time_step=5, method='saits'):
    if method == 'saits':
        ori_floor_df = pd.read_csv(f'../data/fp_sample_{dataset_name}.csv', index_col=0)
    else:
        ori_floor_df = pd.read_csv(f'../data/fp_sample_{dataset_name}.csv', index_col=0)

    ori_floor_df = ori_floor_df.sort_values(by=['ts'], ascending=True).reset_index(drop=True)

    # ori_floor_df = ori_floor_df.sample(frac=1, random_state=2021).reset_index(drop=True)
    ori_floor_df.ts = ori_floor_df.ts.astype(float)/1000
    start_time = ori_floor_df.loc[0, ['ts']].values[0]
    print('start time:', start_time, type(start_time))
    print('window:', window)
    window_thre = start_time + float(window)*60
    ori_floor_df = ori_floor_df.loc[ori_floor_df['ts'] <= window_thre, :]
    print('window data len:', ori_floor_df.shape[0])
    metadata_cols = ['floor', 'x', 'y', 'wp_ts', 'ts', 'path']
    Feature_df = ori_floor_df.drop(columns=metadata_cols, errors='ignore')
    # Ensure features are numeric so downstream np.isnan works safely.
    Feature_df = Feature_df.apply(pd.to_numeric, errors='coerce')
    print('feature length:', Feature_df.shape[1])

    X = Feature_df.values
    if method == 'saits':
        num_samples = X.shape[0] //time_step
        X = X[:num_samples*time_step]
        X = StandardScaler().fit_transform(X)
        X = X.reshape(num_samples, time_step, -1)
        return X
    else:
        return X


def load_ICU_dataset(window=2, method='saits', stream=1):
    data = load_specific_dataset('physionet_2012')  # For datasets in PyPOTS database, PyPOTS will automatically download and extract it.
    X = data['X']
    print(X.columns)
    num_samples = len(X['RecordID'].unique())
    X = X.drop('RecordID', axis=1)
    print('X shape', X.shape)  # X shape (575424, 37)
    print('sum of nan:', np.sum(np.isnan(X)))

    columns = ['RecordID', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
       'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
       'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
       'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
       'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine',
       'WBC', 'Weight', 'pH']

    X = X.to_numpy()
    if method == 'saits':
        X = StandardScaler().fit_transform(X)
        X = X.reshape(num_samples, 48, -1)
        np.random.shuffle(X)
        X = X[:int(num_samples*stream),:window, :]
        return X
    else:
        X = X.reshape(num_samples, 48, -1)
        np.random.shuffle(X)
        X = X[:int(num_samples*stream),:window, :]
        return X.reshape(-1, 37)

# def load_airquality_dataset(window=2, method='saits'):
#     # data = load_specific_dataset('physionet_2012')  # For datasets in PyPOTS database, PyPOTS will automatically download and extract it.
#
#     data = tsdb.load_dataset('beijing_multisite_air_quality')  # select the dataset you need and load it, TSDB will download, extract, and process it automaticallyp
#
#     # [541991 rows x 43 columns],
#     X = data['X']
#     # X.to_csv('../OCW_data/airquality.csv', header=True, index=False)
#     # print(X.columns)
#
#     print('X original:', X.shape)
#
#     columns = ['No', 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2',
#        'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station']
#
#     num_stations = len(X['station'].unique())
#
#     print('num_stations:', num_stations)
#
#     X = X.groupby(['station']).apply(lambda x: x.sort_values(by=['No', 'year', 'month', 'day', 'hour'], ascending=True)).reset_index(drop=True)
#
#     # X.to_csv('../OCW_data/airquality_sorted.csv', header=True, index=False)
#
#     print('X grouped', X.shape)
#     X = X.drop(['No', 'year', 'month', 'day', 'hour', 'wd', 'station'], axis=1)
#
#     # X.to_csv('../OCW_data/airquality_short.csv', header=True, index=False)
#
#     X = X.to_numpy()
#     num_of_channels = X.shape[1]
#
#     if method == 'saits':
#         X = StandardScaler().fit_transform(X)
#         X = X.reshape(num_stations, -1, num_of_channels)
#         print('X shape111', X.shape)
#         X = X[:,:window, :]
#         return X
#     else:
#         X = X.reshape(num_stations, -1, num_of_channels)
#         X = X[:,:window, :]
#         return X.reshape(-1, num_of_channels)


def load_airquality_dataset(window=2, method='saits', stream=1):

    data = tsdb.load_dataset('beijing_multisite_air_quality')  # select the dataset you need and load it, TSDB will download, extract, and process it automaticallyp

    # [541991 rows x 43 columns],
    X = data['X']
    # X.to_csv('../OCW_data/airquality.csv', header=True, index=False)
    # print(X.columns)

    print('X original:', X.shape)

    columns = ['No', 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2',
       'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station']

    print('X grouped', X.shape)
    X = X.drop(['No', 'year', 'month', 'day', 'hour', 'wd', 'station'], axis=1)

    # X.to_csv('../OCW_data/airquality_short.csv', header=True, index=False)

    X = X.to_numpy()
    num_of_channels = X.shape[1]

    if method == 'saits':
        X = StandardScaler().fit_transform(X)
        X = X.reshape(-1, 24, num_of_channels)
        num_samples = X.shape[0]
        np.random.shuffle(X)
        X = X[:int(num_samples*stream),:window, :]
        return X
    else:
        X = X.reshape(-1, 24, num_of_channels)
        num_samples = X.shape[0]
        np.random.shuffle(X)
        X = X[:int(num_samples*stream),:window, :]
        return X.reshape(-1, num_of_channels)


def load_WiFi_dataset_all(dataset_name='KDM', time_step=5, method='saits'):
    if method == 'saits':
        ori_floor_df = pd.read_csv(f'../data/fp_sample_{dataset_name}.csv', index_col=0)
    else:
        ori_floor_df = pd.read_csv(f'./data/fp_sample_{dataset_name}.csv', index_col=0)
    ori_floor_df = ori_floor_df.sort_values(by=['ts'], ascending=True).reset_index(drop=True)

    # ori_floor_df = ori_floor_df.sample(frac=1, random_state=2021).reset_index(drop=True)
    ori_floor_df.ts = ori_floor_df.ts.astype(float)/1000
    start_time = ori_floor_df.loc[0, ['ts']].values[0]
    print('start time:', start_time, type(start_time))
    metadata_cols = ['floor', 'x', 'y', 'wp_ts', 'ts', 'path']
    Feature_df = ori_floor_df.drop(columns=metadata_cols, errors='ignore')
    # Ensure all feature columns are numeric so downstream np.isnan works safely.
    Feature_df = Feature_df.apply(pd.to_numeric, errors='coerce')
    print('feature length:', Feature_df.shape[1])
    # normalize all samples:
    X = Feature_df.values
    if method == 'saits':
        num_samples = X.shape[0] //time_step
        X = X[:num_samples*time_step]
        X = StandardScaler().fit_transform(X)
        X = X.reshape(num_samples, time_step, -1)
        return X
    else:
        return X


def load_ICU_dataset_all(method='saits', stream=1):
    data = load_specific_dataset('physionet_2012')  # For datasets in PyPOTS database, PyPOTS will automatically download and extract it.
    X = data['X']
    print(X.columns)
    num_samples = len(X['RecordID'].unique())
    X = X.drop('RecordID', axis=1)
    print('X shape', X.shape)  # X shape (575424, 37)
    print('sum of nan:', np.sum(np.isnan(X)))

    columns = ['RecordID', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
       'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
       'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
       'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
       'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine',
       'WBC', 'Weight', 'pH']

    X = X.to_numpy()
    if method == 'saits':
        X = StandardScaler().fit_transform(X)
        X = X.reshape(num_samples, 48, -1)
        np.random.shuffle(X)
        X = X[:int(num_samples*stream), :, :]
        X = np.transpose(X, (1,0,2))
        return X
    else:
        X = X.reshape(num_samples, 48, -1)
        np.random.shuffle(X)
        X = X[:int(num_samples*stream), :, :]
        X = np.transpose(X, (1,0,2))

        return X.reshape(-1, 37)


def load_airquality_dataset_all(method='saits', stream=1):

    data = tsdb.load_dataset('beijing_multisite_air_quality')  # select the dataset you need and load it, TSDB will download, extract, and process it automaticallyp

    # [541991 rows x 43 columns],
    X = data['X']
    # X.to_csv('../OCW_data/airquality.csv', header=True, index=False)
    # print(X.columns)

    print('X original:', X.shape)

    columns = ['No', 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2',
       'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station']

    print('X grouped', X.shape)
    X = X.drop(['No', 'year', 'month', 'day', 'hour', 'wd', 'station'], axis=1)

    # X.to_csv('../OCW_data/airquality_short.csv', header=True, index=False)

    X = X.to_numpy()
    num_of_channels = X.shape[1]

    if method == 'saits':
        X = StandardScaler().fit_transform(X)
        X = X.reshape(-1, 24, num_of_channels)
        num_samples = X.shape[0]
        np.random.shuffle(X)
        X = X[:int(num_samples*stream),:, :]
        X = np.transpose(X, (1,0,2))
        return X
    else:
        X = X.reshape(-1, 24, num_of_channels)
        num_samples = X.shape[0]
        np.random.shuffle(X)
        X = X[:int(num_samples*stream),:, :]
        # X = np.transpose(X, (1,0,2))
        return X.reshape(-1, num_of_channels)


def load_weather_dataset(window=2, method='saits', stream=1):
    # Load Vietnam weather dataset from ../data/weather.csv
    df = pd.read_csv('../data/weather.csv')

    # Parse date and ensure proper types
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['date', 'province']).reset_index(drop=True)

    # Map wind direction codes to degrees (center angles)
    wind_dir_map = {
        'N': 0.0,
        'NNE': 22.5,
        'NE': 45.0,
        'ENE': 67.5,
        'E': 90.0,
        'ESE': 112.5,
        'SE': 135.0,
        'SSE': 157.5,
        'S': 180.0,
        'SSW': 202.5,
        'SW': 225.0,
        'WSW': 247.5,
        'W': 270.0,
        'WNW': 292.5,
        'NW': 315.0,
        'NNW': 337.5,
    }

    # Select numeric feature columns and convert
    features = ['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure']
    # Create numeric wind direction column
    df['wind_dir_deg'] = df['wind_d'].map(wind_dir_map)
    features.insert(3, 'wind_dir_deg')

    # Ensure numeric conversion
    for c in features:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Collect per-province sequences, trim to multiples of window
    seqs = []
    for _, g in df.groupby('province'):
        g = g.sort_values('date')
        arr = g[features].to_numpy()
        if arr.shape[0] < window:
            continue
        n = arr.shape[0] // window
        arr = arr[: n * window, :]
        seqs.append(arr)

    if len(seqs) == 0:
        return np.zeros((0, len(features))) if method != 'saits' else np.zeros((0, window, len(features)))

    global_arr = np.vstack(seqs)

    # Fill missing values with column means before scaling
    col_means = np.nanmean(global_arr, axis=0)
    inds = np.where(np.isnan(global_arr))
    if inds[0].size > 0:
        global_arr[inds] = np.take(col_means, inds[1])

    num_of_channels = global_arr.shape[1]

    if method == 'saits':
        global_arr = StandardScaler().fit_transform(global_arr)
        X = global_arr.reshape(-1, window, num_of_channels)
        num_samples = X.shape[0]
        np.random.shuffle(X)
        X = X[: int(num_samples * stream), :window, :]
        return X
    else:
        global_arr = StandardScaler().fit_transform(global_arr)
        X = global_arr.reshape(-1, window, num_of_channels)
        return X.reshape(-1, num_of_channels)


def load_weather_dataset_all(method='saits', stream=1):
    # Return transposed version suitable for _all variants used elsewhere
    X = load_weather_dataset(window=2, method=method, stream=stream)
    if method == 'saits':
        # transpose to (window, num_samples, channels) as some callers expect
        return np.transpose(X, (1, 0, 2))
    else:
        return X


# a = load_airquality_dataset()

# load_ICU_dataset()

# load_WiFi_dataset('KDM')


# WDS: 135 mins
# KDM: 59.9 mins