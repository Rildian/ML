# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# %%
data = pd.read_csv('../Dados/census.csv')

# %%
data

# %%
x_census = data.iloc[::, 0:14].values

# %%
x_census

# %%
y_census = data.iloc[::, 14].values
y_census

# %%
label_enconder_teste = LabelEncoder()

# %%
x_census[:, 1]

# %%
teste = label_enconder_teste.fit_transform(x_census[:, 1].astype(str))
teste

# %%
data

# %%
data.dtypes

# %%
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital_status = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

# %%
x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])         # workclass
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])         # education
x_census[:, 5] = label_encoder_marital_status.fit_transform(x_census[:, 5])    # marital-status
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])        # occupation
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])      # relationship
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])              # race
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])               # sex
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])         # country

# %%
x_census[0]

# %%
x_census2 = x_census.copy()
y_census2 = y_census.copy()

# %%
from sklearn.compose import ColumnTransformer

# %%
one_hot_enconder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')

# %%
x_census2 = one_hot_enconder_census.fit_transform(x_census2).toarray()

# %%
x_census2

# %%
data.shape

# %%
x_census.shape

# %%
x_census2.shape

# %%
scaler_x = StandardScaler()
x_census2 = scaler_x.fit_transform(x_census2)

# %%
x_census2

# %%
from sklearn.model_selection import train_test_split

# %%
x_census_training, x_census_test, y_census_training, y_census_test = train_test_split(
    x_census2, y_census2, test_size=0.15, random_state=0
)
# treinamento, teste, treinamento, teste, x,y

# %%
x_census_training.shape, y_census_training.shape

# %%
x_census_test.shape, y_census_test.shape

# %%
import pickle

# %%
with open('census.pkl', mode='wb') as f:
    pickle.dump([x_census_training, y_census_training, x_census_test, y_census_test], f)


