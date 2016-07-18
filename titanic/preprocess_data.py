# coding:utf-8
# preprocess data
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from conf import conf


def get_unique_map(s):
    m_dict = {}
    for i, v in enumerate(np.unique(s)):
        m_dict[v] = i
    return m_dict

def preprocess_data(df, label = True):
    cols = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    df2 = df.copy()
    df2[['Age','Fare']] = df[['Age','Fare']].fillna(df[['Age','Fare']].mean())
    df2['Sex'] = df.Sex.map({'male':1, 'female':0})
    df2['Embarked'] = df.Embarked.map(get_unique_map(df.Embarked))

    if label:
        return df2[cols], df2.Survived
    return df2[cols]

if __name__ == '__main__':
    train = pd.read_csv(conf.train)
