import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt


train_1999 = pd.read_csv('./data/ace/ace_1999.csv')
print("train shape: ", train_1999.shape)

test = pd.read_csv('./data/test.csv')
print("test shape", test.shape)

label = pd.read_csv('./data/kp_index.csv')
print("label shape", label.shape)


def process_na(dataset):
    # global dataset
    # print(dataset.columns)
    for feature in dataset.columns:
        dataset[feature] = dataset[feature].map(lambda x: np.nan if x < -9999 else x)

    return dataset

train_1999 = process_na(train_1999)

time = np.linspace(1999, 2013, train_1999.shape[0])
Np = train_1999['Np']
Tp = train_1999['Tp']
Vp = train_1999['Vp']


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(time, Np)
ax.set_ylabel('Np')
ax.set_xlabel('year')
plt.title('Np vs. Year')
plt.grid(True)
plt.savefig('./eda_output_img/Np_time.png')
plt.show()


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(time, Tp)
ax.set_ylabel('Tp')
ax.set_xlabel('year')
plt.title('Tp vs. Year')
plt.grid(True)
plt.savefig('./eda_output_img/Tp_time.png')
plt.show()


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(time, Vp)
ax.set_ylabel('Vp')
ax.set_xlabel('year')
plt.title('Vp vs. Year')
plt.grid(True)
plt.savefig('./eda_output_img/Vp_time.png')
plt.show()
