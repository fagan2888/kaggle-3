# coding:utf-8
# config
import pandas as pd
conf = {
    'train' : 'train.csv',
    'test' : 'test.csv',
    'processed_train' : 'processed_train.csv',
    'processed_test' : 'processed_test.csv',
    'random_state' : 314,
    'test_size' : 0.4
}
conf = pd.Series(conf)
