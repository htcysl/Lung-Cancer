import pandas as pd

df = pd.read_excel("cancer patient data sets.xlsx")
df.to_csv ("Test.csv", index = None, header=True)
print(df)