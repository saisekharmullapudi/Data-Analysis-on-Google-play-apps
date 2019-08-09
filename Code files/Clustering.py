
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.tools.plotting import parallel_coordinates 
from sklearn.utils import shuffle


# In[3]:


apps = pd.read_csv("googleplaystore.csv")
apps.dropna(inplace = True)
apps.reset_index(drop=True,inplace=True)


# In[4]:


catval = apps['Category'].unique()
dict1 = {}
i = 0
for each in catval:
    dict1[each] = i
    i = i+1


# In[5]:


apps["CategoryValues"] = apps["Category"].map(dict1).astype(int)


# In[6]:


apps['Installs'] = [int(i[:-1].replace(',','')) for i in apps['Installs']]


# In[7]:


installs = apps["Installs"]
Installs = np.array(installs)
# print(installs)
def znormalization(ts):    
    mean = ts.mean(axis = 0)
    std = ts.std(axis = 0)
    return (ts - mean) / std
normalized = znormalization(installs)


# In[8]:


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


# In[9]:


apps['Size'] = apps['Size'].map(convertsize)
def typetobool(type):
    if type == "Free":
        return 0
    else:
        return 1
apps['Type'] = apps['Type'].map(typetobool)


# In[10]:


#converting the Generes into numeric values numbers
genreval = apps['Genres'].unique()
# print(genreval)
genresdict = {}
j = 0
for each in genreval: 
    genresdict[each] = j
    j = j + 1
# print(genresdict)


# In[11]:


apps['GenreValues'] = apps['Genres'].map(genresdict).astype(int)
# print(apps['Genres'])


# In[12]:


new_df = apps.filter(['CategoryValues','Rating','Reviews','Type','GenreValues'],axis=1)
new_df = shuffle(new_df)
new_df=new_df.ix[:,'Rating':]
y = apps['CategoryValues']


# In[13]:


lda = LDA(n_components=2)


# In[14]:


lda_transformed = pd.DataFrame(lda.fit_transform(new_df,y))


# In[17]:


plt.scatter(lda_transformed[y==1][0], lda_transformed[y==1][1], c='red')
plt.scatter(lda_transformed[y==2][0], lda_transformed[y==2][1], c='blue')
plt.scatter(lda_transformed[y==3][0], lda_transformed[y==3][1], c='lightgreen')
plt.scatter(lda_transformed[y==4][0], lda_transformed[y==4][1],  c='orange')
plt.scatter(lda_transformed[y==5][0], lda_transformed[y==5][1], c='black')
plt.scatter(lda_transformed[y==6][0], lda_transformed[y==6][1],  c='purple')
plt.scatter(lda_transformed[y==7][0], lda_transformed[y==7][1],  c='orange')
plt.scatter(lda_transformed[y==8][0], lda_transformed[y==8][1],  c='brown')
plt.scatter(lda_transformed[y==9][0], lda_transformed[y==9][1],  c='gray')
plt.scatter(lda_transformed[y==10][0], lda_transformed[y==10][1],  c='magenta')
plt.scatter(lda_transformed[y==11][0], lda_transformed[y==11][1], c='maroon')
plt.scatter(lda_transformed[y==12][0], lda_transformed[y==12][1],  c='navy')

# Display legend and show plot

plt.show()

