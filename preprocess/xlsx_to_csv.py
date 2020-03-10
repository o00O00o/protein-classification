import xlrd
import pandas as pd

path = "ProCla/data/allData.xlsx"
excel = xlrd.open_workbook(path)
sheet = excel.sheet_by_index(0)
rowNum = sheet.nrows
dataCol = []
labCol = []
lenCol = []

for i in range(1, rowNum):
    data = sheet.cell_value(i, 14)
    label = sheet.cell_value(i, 13)
    if (data != ''):
        length = len(data)
        if (length < 1008 and length > 100):
            dataCol.append(data)
            labCol.append(label)
            lenCol.append(length)

df = pd.DataFrame({'Data': dataCol, 'Label': labCol, 'Length': lenCol})
df.to_csv('ProCla/data/part_data.csv')
