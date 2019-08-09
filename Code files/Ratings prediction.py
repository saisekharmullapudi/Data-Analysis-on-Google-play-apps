
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[14]:


path1 = "googleplaystore.csv"
apps = pd.read_csv(path1)
apps.dropna(inplace=True)


# In[15]:


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
# print(dict1)


# In[16]:


apps["CategoryValues"] = apps["Category"].map(dict1).astype(int)
# print(apps["CategoryValues"])


# In[17]:



#normalizing the install column in the csv file
apps['Installs'] = [int(i[:-1].replace(',','')) for i in apps['Installs']]



# In[18]:


#normalizing the downloads since they are distributed non-uniformly
installs = apps["Installs"]
installs = np.array(installs)
# print(installs)
def znormalization(ts):    
    mean = ts.mean(axis = 0)
    std = ts.std(axis = 0)
    return (ts - mean) / std
normalized = znormalization(installs)
print(normalized)


# In[220]:


# pre-processing the size column in the features 

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
    


# In[221]:


apps['Size'] = apps['Size'].map(convertsize)


# In[222]:


# print(apps['Size'])


# In[223]:


#converting the type column to boolean i.e if free make it to 0 if paid make it to 1
def typetobool(type):
    if type == "Free":
        return 0
    else:
        return 1


# In[224]:


apps['Type'] = apps['Type'].map(typetobool)


# In[225]:


# print(apps['Type'])


# In[226]:


#cleaning the price column in the csv, prices are prefix with a "$" sign in the begining


# In[229]:


def convertprice(price):
    if "$" in price:
        price1 = price[1:]
        return float(price1)
    else:
        return float(price)


# In[230]:


apps['Price'] = apps['Price'].map(convertprice)


# In[231]:


# print(apps['Price'])


# In[232]:


#converting the Generes into numeric values numbers
genreval = apps['Genres'].unique()
# print(genreval)
genresdict = {}
j = 0
for each in genreval: 
    genresdict[each] = j
    j = j + 1
# print(genresdict)


# In[233]:


apps['GenreValues'] = apps['Genres'].map(genresdict).astype(int)
# print(apps['Genres'])


# In[240]:


# print(apps['CategoryValues'])


# In[235]:


new_apps = apps[['CategoryValues','Installs','Size','Type','Price','GenreValues']].copy()


# In[236]:


y = apps['Rating']


# In[237]:


x_train,x_test,y_train,y_test=train_test_split(new_apps,y,test_size=0.20)


# In[238]:


model = LinearRegression()
model.fit(x_train,y_train)
Results = model.predict(x_test)


# In[255]:


print(Results[520])


# In[250]:


def meanError(y1,y2):
    sumval = 0
    i = 0
    while i < len(y1):
        sumval += abs(y1[i] - y2[i])
        i = i+1
    return sumval/len(y1)


# In[251]:


def MSE(y1,y2):
    SE = 0
    i = 0
    while i < len(y1):
        SE += (y1[i]-y2[i])**2
        i = i+1
    return SE/len(y1)


# In[252]:


# print("The mean error between the actual values and the predicted values are",meanError(y_test,Results))

