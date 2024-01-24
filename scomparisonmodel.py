import pandas as pd


df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
print(df1)  
print(df2)

result1 = df1.apply(tuple, 1).sin(df2.apply(tuple,1))
print(result1)


df1.merge(df3, indicators=True, how = 'outer').loc[lambda v: v['_merge'] !='both']