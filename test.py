import pandas as pd 

df = pd.read_csv('labels_final.csv')
names = df['image'].unique()
print(len(names))