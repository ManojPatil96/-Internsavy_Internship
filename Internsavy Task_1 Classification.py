#!/usr/bin/env python
# coding: utf-8

# # Data Science Internship by Internsavy

# Name - Manoj Patil

# Task 1 
# 
# Use classification technique for
# prediction of Graduate Admissions 
# from an Indian perspective.
# 
# 

# In[1]:


# Importing required libraries 


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# loading data set

df=pd.read_csv("IS_Admission_Predict.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.columns


# In[6]:


df.isna().sum()


# In[7]:


df.info()


# In[8]:


df.describe()


# # Determine label and Features (Y&X)

# In[9]:


df.columns


# In[10]:


features=['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Chance of Admit ']


# In[11]:


x=df[features]
x


# In[12]:


y=df.Research
y


# # Splitting the dataset into training and Testing

# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[14]:


x_train


# In[15]:


x_test


# In[16]:


y_train


# In[17]:


y_test


# # Train the Algorithm

# In[18]:


# Using decision tree model


from sklearn.tree import DecisionTreeClassifier


# In[19]:


dt=DecisionTreeClassifier()


# In[20]:


dt.fit(x_train,y_train)


# # Predecting the test data set(x_test)

# In[21]:


y_pred=dt.predict(x_test) 


# In[22]:


y_pred# model predicted


# In[23]:


y_test


# # Evaluating the Performance of Model

# In[24]:


import matplotlib.pyplot as plt
from sklearn import metrics


confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
confusion_matrix = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=[False,True])
confusion_matrix.plot()
plt.show()


# In[25]:


#Accuracy

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)*100


# In[26]:


#Precision

from sklearn.metrics import precision_score
precision_score(y_test,y_pred,average=None)*100


# In[27]:


#Recall


from sklearn.metrics import recall_score

recall_score(y_test,y_pred,average=None)*100


# In[28]:


# F1 Score

from sklearn.metrics import f1_score
f1_score(y_test,y_pred,average=None)*100 


# In[ ]:




