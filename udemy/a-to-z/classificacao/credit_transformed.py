# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv('../Dados/credit_data.csv')

# %%
value = df.query("age > 0")["age"].mean()
df.loc[df['age'] < 0, 'age'] = value

# %%
df2 = df.copy()
np.where(pd.isnull(df2))
# [28, 30, 31]

df2.drop(index=[28, 30, 31], axis=0, inplace=True) # dropar essas linhas

# %%
value = df2['age'].mean()

# %%
df.fillna(value=value, inplace=True)

# %%
df2.isna().sum()

# %%
x_credit = df.iloc[::, 1:4].values
y_credit = df.iloc[::, 4]

# %%
scaler_credit = StandardScaler()

# %%
x_credit = scaler_credit.fit_transform(x_credit)
x_credit

# %%
from sklearn.model_selection import train_test_split

# %%
x_credit_training, x_credit_test, y_credit_training, y_credit_test = train_test_split(
    x_credit, y_credit, test_size = 0.25, random_state=0
)

# %%
x_credit.shape

# %%
x_credit_test.shape

# %%
x_credit_training.shape

# %%
import pickle

# %%
with open('credit.pkl', mode='wb') as f:
    pickle.dump([x_credit_training, y_credit_training, 
                 x_credit_test, y_credit_test], f)


