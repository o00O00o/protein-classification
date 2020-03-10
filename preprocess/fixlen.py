import pandas as pd
import numpy as np

df = pd.read_csv("ProCla/data/part_data.csv")

dataCol = []
labCol = []

for index, row in df.iterrows():
    data = row['Data']
    label = row['Label']
    length = row['Length']

    if (length >= 100 and length < 250):
        data = data + data + data + data + data
        data = data[0:500]

    if (length >= 250 and length < 500):
        data = data + data
        data = data[0:500]

    if (length > 500):
        ranNum = np.random.randint(250, length - 250)
        data = data[ranNum - 250:ranNum + 250]

    dataCol.append(data)
    if (label == "Protein-DNA"):
        labCol.append(0)
    else:
        labCol.append(1)

df = pd.DataFrame({'Data': dataCol, 'Label': labCol})
print(df.head())
print(len(df.at[4, 'Data']))
df.to_csv('ProCla/data/fixlen_data.csv')
