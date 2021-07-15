#!/usr/bin/env python
# coding: utf-8

# ### Import libraries and data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[205]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(train_df.head())


# In[111]:


test_df.head()


# In[112]:


print(train_df.isnull().sum())
print(test_df.isnull().sum())


# In[113]:


print(train_df.info())
print(test_df.info())


# ### Data Visualization

# In[114]:


sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='plasma')


# In[115]:


sns.heatmap(test_df.isnull(), yticklabels=False, cbar=False, cmap='plasma')


# In[116]:


train_df['Fare'].hist(bins=40, figsize=(10,6))


# In[117]:


test_df['Fare'].hist(bins=40, figsize=(10,6))


# In[118]:


sns.set_style("whitegrid")
sns.countplot(x='Survived', data=train_df, hue='Sex')


# In[119]:


sns.countplot(x='Survived', data=train_df, hue='Pclass')


# In[120]:


sns.distplot(train_df['Age'].dropna(), kde=False, bins=25)


# In[121]:


sns.countplot(x='SibSp', data=train_df)


# ### Data Cleaning and preprocessing

# In[122]:


plt.figure(figsize=(10, 8))
sns.boxplot(x='Pclass', y='Age', data=train_df)


# In[123]:


plt.figure(figsize=(10, 8))
sns.boxplot(x='Pclass', y='Age', data=test_df)


# In[128]:


def find_age_train(colmuns):
    Age = colmuns[0]
    Pclass = colmuns[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[206]:


def find_age_test(colmuns):
    Age = colmuns[0]
    Pclass = colmuns[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 42
        elif Pclass == 2:
            return 27
        else:
            return 24
    else:
        return Age


# In[130]:


train_df['Age'] = train_df[['Age', 'Pclass']].apply(find_age_train, axis=1)


# In[207]:


test_df['Age'] = test_df[['Age', 'Pclass']].apply(find_age_test, axis=1)


# In[132]:


sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='plasma')


# In[208]:


sns.heatmap(test_df.isnull(), yticklabels=False, cbar=False, cmap='plasma')


# In[209]:


train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)


# In[210]:


train_df.dropna(inplace=True)
# test_df.dropna(inplace=True)


# In[211]:


sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='plasma')


# In[212]:


sns.heatmap(test_df.isnull(), yticklabels=False, cbar=False, cmap='plasma')


# In[138]:


sex_train = pd.get_dummies(train_df['Sex'], drop_first=True)
sex_train.head()


# In[213]:


sex_test = pd.get_dummies(test_df['Sex'], drop_first=True)
sex_test.head()


# In[140]:


embark_train = pd.get_dummies(train_df['Embarked'], drop_first=True)
embark_train.head()


# In[214]:


embark_test = pd.get_dummies(test_df['Embarked'], drop_first=True)
embark_test.head()


# In[142]:


train = pd.concat([train_df, sex_train, embark_train], axis=1)
train.head()


# In[215]:


test = pd.concat([test_df, sex_test, embark_test], axis=1)
test.head()


# In[144]:


train.drop(['Sex', 'Name', 'Embarked', 'Ticket'], axis=1, inplace=True)
train.head()


# In[216]:


test.drop(['Sex', 'Name', 'Embarked', 'Ticket'], axis=1, inplace=True)
test.head()


# In[146]:


train.drop('PassengerId', axis=1, inplace=True)
train.head()


# In[217]:


test.drop('PassengerId', axis=1, inplace=True)
test.head()


# In[218]:


print(train.info())
print(test.info())


# ### Training and Evaluation

# In[149]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[262]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[263]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[264]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[265]:


X_test.max(axis=0)


# ### LogisticRegression Model

# In[266]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[267]:


lr.fit(X_train, y_train)


# In[268]:


preds = lr.predict(X_test)
preds


# In[269]:


from sklearn.metrics import classification_report, confusion_matrix


# In[270]:


print(classification_report(y_test, preds))


# In[271]:


print(confusion_matrix(y_test, preds))


# ### RandomForestClassifier Model

# In[272]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()


# In[273]:


rf.fit(X_train, y_train)


# In[274]:


rf_preds = rf.predict(X_test)
rf_preds


# In[275]:


print(classification_report(y_test, rf_preds))


# ### DecisionTreeClassifier Model

# In[276]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[277]:


dt.fit(X_train, y_train)


# In[278]:


dt_pred = dt.predict(X_test)
dt_pred


# In[279]:


print(classification_report(y_test, dt_pred))


# ### GaussianNB Model

# In[280]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()


# In[281]:


gnb.fit(X_train, y_train)


# In[282]:


gnb_preds = gnb.predict(X_test)
gnb_preds


# In[283]:


print(classification_report(y_test, gnb_preds))


# ### KNeighborsClassifier Model

# In[284]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[285]:


knn.fit(X_train, y_train)


# In[286]:


knn_preds = knn.predict(X_test)
knn_preds


# In[287]:


print(classification_report(y_test, knn_preds))


# In[288]:


test['Fare'].mean()


# In[289]:


test['Fare'].fillna(test['Fare'].mean(), inplace=True)
test.info()


# In[296]:


test = scaler.transform(test)
test.min(axis=0)


# In[297]:


submit_predictions = lr.predict(test)
submit_predictions


# In[298]:


submit_predictions_df = pd.DataFrame(submit_predictions, columns=['Survived'])
submit_predictions_df.shape


# In[299]:


test_passenderId = test_df['PassengerId'].reset_index()
test_passenderId.shape


# In[300]:


submission_data = pd.concat([test_passenderId['PassengerId'], submit_predictions_df], axis=1)


# In[301]:


submission_data.shape


# In[302]:


submission_data.to_csv('gender_submission.csv', index=False)


# In[ ]:




