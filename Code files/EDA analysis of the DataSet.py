
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


path1 = "googleplaystore.csv"
apps = pd.read_csv(path1)
apps.dropna(inplace=True)


# In[3]:


#data pre-processing
#converting the category into numeric value
#predicting the ratings of the apps by using multivariate linear regression , features to consider are category, size, installs,
# type, price , genre

catval = apps['Category'].unique()
dict1 = {}
i = 0
for each in catval:
    dict1[each] = i
    i = i+1


# In[4]:


apps["CategoryValues"] = apps["Category"].map(dict1).astype(int)


# In[5]:


apps['Installs'] = [int(i[:-1].replace(',','')) for i in apps['Installs']]


# In[14]:


#normalizing the downloads since they are distributed non-uniformly
installs = apps["Installs"]
Installs = np.array(installs)
# print(installs)
def znormalization(ts):    
    mean = ts.mean(axis = 0)
    std = ts.std(axis = 0)
    return (ts - mean) / std
normalized = znormalization(installs)


# In[15]:


def convertsize(size1):
    if 'M' in size1:
        size = size1[:-1]
        size = float(size)*1024
        return size
    elif 'K' in size1:
        size = size1[:-1]
        size = float(size)*1
        return size
    else:
        return 1000


# In[53]:


apps['Size'] = apps['Size'].map(convertsize)
def typetobool(type):
    if type == "Free":
        return 0
    else:
        return 1
apps['Type'] = apps['Type'].map(typetobool)


# In[60]:


def convertprice(price):
    if "$" in price:
        price1 = price[1:]
        return float(price1)
    else:
        return float(price)


# In[61]:


apps['Price'] = apps['Price'].map(convertprice)


# In[54]:


#reviews.head()


# In[62]:


genreval = apps['Genres'].unique()
# print(genreval)
genresdict = {}
j = 0
for each in genreval: 
    genresdict[each] = j
    j = j + 1
# print(genresdict)


# In[55]:


#distribution of data in the android play store based on category of apps


# In[63]:


apps['GenreValues'] = apps['Genres'].map(genresdict).astype(int)


# In[4]:


fig, ax = plt.subplots(figsize=(12,8))
ax = sns.countplot(x=apps['Category'])
plt.xticks(rotation=90)
plt.show()


# In[6]:


genresString = apps["Genres"]
fig,ax = plt.subplots(figsize=(25,10))
plt.scatter(x=genresString,y=apps["Rating"],color="green",marker="o")
plt.xticks(rotation=90)
plt.grid()
plt.show()


# In[74]:


#1 Types of payments by rating and comments
plt.figure(figsize=(24,8))
sns.boxplot(x=apps.Installs,y=apps.Rating,hue=apps.Type,palette="PRGn")
plt.title("BOXPLOT",color="red",fontsize=15)
plt.show()


# In[66]:


fig,ax = plt.subplots(figsize=(8,7))
ax = sns.heatmap(apps.corr(), annot=True,linewidths=.5,fmt='.1f')
plt.show()


# In[19]:




