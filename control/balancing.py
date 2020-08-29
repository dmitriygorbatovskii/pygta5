import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('/home/da/PycharmProjects/pythonProject/pygta5_yv5/control/training_data.npy', allow_pickle=True)

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []
backs = []
waits =[]

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append(np.array([img,choice], dtype=object))
    elif choice == [0,1,0]:
        forwards.append(np.array([img,choice], dtype=object))
    elif choice == [0,0,1]:
        rights.append(np.array([img,choice], dtype=object))
        print('no matches')


forwards = forwards[:len(lefts)]
forwards = forwards[:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]



final_data = forwards + lefts + rights
shuffle(final_data)

np.save('training_data1.npy', final_data, allow_pickle=True)

train_data = np.load('/home/da/PycharmProjects/pythonProject/pygta5_yv5/control/training_data.npy', allow_pickle=True)

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

