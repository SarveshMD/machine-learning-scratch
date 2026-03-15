import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

df.drop(axis=0, columns=['id', 'Unnamed: 32'], inplace=True)
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)

df = df.sample(frac=1, random_state=42, ignore_index=True)

df_train = df.iloc[:455] # 455: ~80% of 569
df_test = df.iloc[455:] # 114: ~20% of 569

df_X = df_train.drop(axis=0, columns='diagnosis')
df_Y = df_train['diagnosis']

df_X_means = df_X.mean(axis=0)
df_X_sd = np.sqrt(df_X.var(axis=0))
df_X = (df_X - df_X_means)/df_X_sd

df_test_X = df_test.drop(axis=0, columns='diagnosis')
df_test_Y = df_test['diagnosis']

df_test_X_means = df_test_X.mean(axis=0)
df_test_X_sd = np.sqrt(df_test_X.var(axis=0))
df_test_X = (df_test_X - df_test_X_means)/df_test_X_sd

# Final Data: df_X, df_Y, df_test_X, df_test_Y
# NP Arrays: Xtrain, Ytrain, Xtest, Ytest

Xtrain = np.array(df_X)
Ytrain = np.array(df_Y).reshape((455,1))

Xtest = np.array(df_test_X)
Ytest = np.array(df_test_Y).reshape((114,1))
