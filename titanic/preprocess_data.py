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

def gettitle(x):
    import re
    r = re.compile(r'^.+?,(.+\.).+$')
    m = r.findall(x)
    if m:
        return m[0]
    return ""

def preprocess_data(df, label = True):
    cols = ["Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked","Child", "Adult", "Title", "Miss", "Mrs", "Mr", \
            "FammliySize", "Single", "MidSize", "LargeSize"]
    df2 = df.copy()
    df2['Child'] = (df.Age < 18).astype(int)
    df2['Adult'] = (df.Age >=18).astype(int)
    df2['Title'] = df.Name.map(gettitle)
    df2['Title'] = df2.Title.map(get_unique_map(df2.Title))
    df2['Miss'] = df2.Title.map(lambda x: x=='Miss.')
    df2['Mrs'] = df2.Title.map(lambda x: x=='Mrs.')
    df2['Mr'] = df2.Title.map(lambda x: x=='Mr.')
    df2['FammliySize'] = df.SibSp + df.Parch + 1
    df2['Single'] = df2.FammliySize == 1
    df2['MidSize'] = (df2.FammliySize > 1) & (df2.FammliySize < 5)
    df2['LargeSize']= (df2.FammliySize > 4)

    df2[['Age','Fare']] = df[['Age','Fare']].fillna(df[['Age','Fare']].mean())

    df2['Sex'] = df.Sex.map({'male':1, 'female':0})
    df2['Embarked'] = df.Embarked.map(get_unique_map(df.Embarked))
    df2['Cabin'] = df.Cabin.fillna('Z').map(lambda x: x[0]).map(get_unique_map(list('ABCDEFGTZ')))

    if label:
        return df2[cols], df2.Survived
    return df2[cols]

if __name__ == '__main__':
    train = pd.read_csv(conf.train)
