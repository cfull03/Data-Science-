#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


# In[2]:


data = '/Users/christianfullerton/Desktop/Python Workspace/Python Datasets/telecom_churn_clean.csv'

df = pd.read_csv(data)
df.head()


# In[3]:


print(df.shape[0])


# In[4]:


X = df.drop('total_day_charge', axis=1)
y = df['total_day_charge']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


lasso_model = Lasso(alpha = 0.01)
lasso_model.fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)


# In[7]:


ridge_model = Ridge(alpha = 0.01)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)


# In[8]:


linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_pred = linear_model.predict(X_test_scaled)


# In[9]:


lasso_score = lasso_model.score(X_test_scaled, y_test)
ridge_score = ridge_model.score(X_test_scaled, y_test)
linear_score = linear_model.score(X_test_scaled, y_test)


# In[10]:


print(f"Lasso Model Score: {lasso_score}")
print(f"Ridge Model Score: {ridge_score}")
print(f"LinearRegression Model Score: {linear_score}")


# In[11]:


print('Mean Squared Error Scores:')
print(f"Lasso MSE: {mean_squared_error(y_test, lasso_pred):.4f}")
print(f"Ridge MSE: {mean_squared_error(y_test, ridge_pred):.4f}")
print(f"Linear Regression MSE: {mean_squared_error(y_test, linear_pred):.4f}")


# In[13]:


print("Cross Valiation Scores")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = 'r2'


lasso_cv = cross_val_score(lasso_model, X_train_scaled, y_train, cv=kf, scoring=scoring)
ridge_cv = cross_val_score(ridge_model, X_train_scaled, y_train, cv=kf, scoring=scoring)
linear_cv = cross_val_score(linear_model, X_train_scaled, y_train, cv=kf, scoring=scoring)

print(f"Lasso CV MSE Scores: {lasso_cv}")
print(f"Ridge CV MSE Scores: {ridge_cv}")
print(f"Linear CV MSE Scores: {linear_cv}")


# In[ ]:




