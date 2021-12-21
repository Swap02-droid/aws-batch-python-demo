
# In[24]:


import numpy as np
import pandas as pd


# In[25]:


dataset=pd.read_csv('train.csv')
print(dataset.isnull().sum())


# #### Cleaning Data

# In[26]:


def for_cleaning(cols):
    age=cols[0]
    pclass=cols[1]
    
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        else:
            return 24
    else:
        return age


# In[27]:


dataset['Age']=dataset[['Age', 'Pclass']].apply(for_cleaning, axis=1)


# In[28]:


dataset.drop('Cabin', axis=1, inplace=True)


# In[29]:


dataset.dropna(inplace=True)


# In[30]:


embarked= pd.get_dummies(dataset['Embarked'], drop_first=True)
sex= pd.get_dummies(dataset['Sex'], drop_first=True)


# In[31]:


dataset.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)


# In[32]:


dataset=pd.concat([dataset, embarked, sex], axis=1)


# In[33]:


dataset.head()


# #### Building a model

# In[34]:


x=dataset.drop('Survived', axis=1)
y=dataset['Survived']


# In[35]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.30, random_state=101)


# In[36]:


from sklearn.linear_model import LogisticRegression


# In[37]:


lsg=LogisticRegression()
lsg.fit(x_train, y_train)


# In[38]:


prediction=lsg.predict(x_test)
print(prediction)
print(len(prediction))


# In[39]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cnm=confusion_matrix(y_test, prediction)
accuracy=accuracy_score(y_test, prediction)

print(cnm)
print(accuracy)
# x_test['Survived']=prediction
# print(x_test)
# x_test.to_csv('Survived.csv')