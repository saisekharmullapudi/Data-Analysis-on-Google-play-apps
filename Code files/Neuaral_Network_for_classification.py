
# coding: utf-8

# In[146]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle


# In[ ]:



# In[130]:


apps = pd.read_csv("googleplaystore.csv")
apps.dropna(inplace=True)
apps.reset_index(drop=True,inplace=True)


# In[131]:


catval = apps['Category'].unique()
dict1 = {}
i = 0
for each in catval:
    dict1[each] = i
    i = i+1


# In[132]:


apps["CategoryValues"] = apps["Category"].map(dict1).astype(int)


# In[133]:


apps['Installs'] = [int(i[:-1].replace(',','')) for i in apps['Installs']]


# In[134]:


#normalizing the downloads since they are distributed non-uniformly
installs = apps["Installs"]
Installs = np.array(installs)
# print(installs)
def znormalization(ts):    
    mean = ts.mean(axis = 0)
    std = ts.std(axis = 0)
    return (ts - mean) / std
normalized = znormalization(installs)


# In[135]:


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


# In[136]:


apps['Size'] = apps['Size'].map(convertsize)
def typetobool(type):
    if type == "Free":
        return 0
    else:
        return 1
apps['Type'] = apps['Type'].map(typetobool)


# In[137]:


def convertprice(price):
    if "$" in price:
        price1 = price[1:]
        return float(price1)
    else:
        return float(price)


# In[138]:


apps['Price'] = apps['Price'].map(convertprice)


# In[139]:


genreval = apps['Genres'].unique()
# print(genreval)
genresdict = {}
j = 0
for each in genreval: 
    genresdict[each] = j
    j = j + 1
# print(genresdict)


# In[140]:


apps['GenreValues'] = apps['Genres'].map(genresdict).astype(int)


# In[1]:



new_df = apps.filter(['CategoryValues','Installs','Size','Type','Price','GenreValues'],axis=1)
new_df = shuffle(new_df)


# In[162]:


print(new_df.shape)


# In[163]:


#divided the data into features and labels
X = new_df.filter(['Installs','Size','Type','Price'])
Y = new_df['CategoryValues']


# X = np.random.shuffle(X)


# In[176]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print(Y_test.shape)


# In[165]:


#standardize the scaling so that we can use the same fitted method to transform/scale test data


# In[166]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[167]:


#initializg a neural network
classifier = Sequential()


# In[168]:


#adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=4))
#adding a second layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


# In[169]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[170]:


#Fitting our model
classifier.fit(X_train,Y_train,batch_size=10,nb_epoch=100)


# In[173]:


y_pred = classifier.predict(X_test)

print(y_pred.shape)

