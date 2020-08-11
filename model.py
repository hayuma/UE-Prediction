from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
dataset = pd.read_csv('CA.csv')
x = dataset.iloc[:, :5]
y = dataset.iloc[:, -1]
result = LinearRegression()
result.fit(x, y)
pickle.dump(result, open('model.pkl','wb'))