# read in 3 prediction file and find the mean of them 

import pandas as pd 

DATA_PATH = './S4E5 Regression with a Flood Prediction Dataset//data/'
ORIGINAL_DATA_PATH = DATA_PATH
SUBMISSIONS_PATH = './S4E5 Regression with a Flood Prediction Dataset/submissions/'
TEMP_PATH = './S4E5 Regression with a Flood Prediction Dataset/temp/'

mlp = pd.read_csv(SUBMISSIONS_PATH + 'mlp3.csv', index_col='id')
cnn = pd.read_csv(SUBMISSIONS_PATH + 'cnn3.csv', index_col='id')
autogluon = pd.read_csv(SUBMISSIONS_PATH + 'autogluon_gpu_1.csv', index_col='id')

mean = (mlp + cnn + autogluon) / 3
mean.to_csv(SUBMISSIONS_PATH + 'mean_of_mlp_cnn_autogluon_1.csv')