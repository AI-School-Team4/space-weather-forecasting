import pandas as pd
import numpy as np
import sklearn

def status(feature):
    print('Processing', feature, ': ok')

# train -> 0 ~ 492750
# test -> 492751 ~
# def combining_data():
#     train_1999 = pd.read_csv('./data/ace/ace_1999.csv')
#     print(train_1999.shape)
#
#     test = pd.read_csv('./data/test.csv')
#     print(test.shape)
#
#     # concat train and test
#     # dataset_1999 = train_1999.append(test)
#     dataset = pd.concat(objs=[train_1999, test], axis=0).reset_index(drop=True)
#     status('combining_data')
#     return dataset
#
# dataset = combining_data()

train_1999 = pd.read_csv('./data/ace/ace_1999.csv')
print(train_1999.shape)

test = pd.read_csv('./data/test.csv')
print(test.shape)

label = pd.read_csv('./data/kp_index.csv')
print(label.head)

def process_na(dataset):
    # global dataset
    # print(dataset.columns)
    for feature in dataset.columns:
        dataset[feature] = dataset[feature].map(lambda x: np.nan if x < -9999 else x)

    status('-9999 to NaN')
    return dataset

train_1999 = process_na(train_1999)
test = process_na(test)

# def minutes():
#     global dataset
#     dataset['min'] = dataset.apply(lambda dataset['hr']==)

def ThreeHours(dataset):
    dataset['hr'] = dataset['hr'].map(lambda x: (x//3)*3)

    status('three hours interval')
    return dataset

train_1999 = ThreeHours(train_1999)
test = ThreeHours(test)

def groupByThree(dataset):
    dataset = dataset.groupby(['doy', 'hr']).mean()
    dataset.drop(['min'], inplace=True, axis=1)
    status('grouped')
    return dataset

train_1999 = groupByThree(train_1999)
test = groupByThree(test)

print(train_1999)
print(test)